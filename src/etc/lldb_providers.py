import sys

from lldb import SBValue, SBData, SBError, eBasicTypeLong, eBasicTypeUnsignedLong, \
    eBasicTypeUnsignedChar

# from lldb.formatters import Logger

####################################################################################################
# This file contains two kinds of pretty-printers: summary and synthetic.
#
# Important classes from LLDB module:
#   SBValue: the value of a variable, a register, or an expression
#   SBType:  the data type; each SBValue has a corresponding SBType
#
# Summary provider is a function with the type `(SBValue, dict) -> str`.
#   The first parameter is the object encapsulating the actual variable being displayed;
#   The second parameter is an internal support parameter used by LLDB, and you should not touch it.
#
# Synthetic children is the way to provide a children-based representation of the object's value.
# Synthetic provider is a class that implements the following interface:
#
#     class SyntheticChildrenProvider:
#         def __init__(self, SBValue, dict)
#         def num_children(self)
#         def get_child_index(self, str)
#         def get_child_at_index(self, int)
#         def update(self)
#         def has_children(self)
#         def get_value(self)
#
#
# You can find more information and examples here:
#   1. https://lldb.llvm.org/varformats.html
#   2. https://lldb.llvm.org/python-reference.html
#   3. https://lldb.llvm.org/python_reference/lldb.formatters.cpp.libcxx-pysrc.html
#   4. https://github.com/llvm-mirror/lldb/tree/master/examples/summaries/cocoa
####################################################################################################

PY3 = sys.version_info[0] == 3


class ValueBuilder:
    def __init__(self, valobj):
        # type: (SBValue) -> ValueBuilder
        self.valobj = valobj
        process = valobj.GetProcess()
        self.endianness = process.GetByteOrder()
        self.pointer_size = process.GetAddressByteSize()

    def from_int(self, name, value):
        # type: (str, int) -> SBValue
        type = self.valobj.GetType().GetBasicType(eBasicTypeLong)
        data = SBData.CreateDataFromSInt64Array(self.endianness, self.pointer_size, [value])
        return self.valobj.CreateValueFromData(name, data, type)

    def from_uint(self, name, value):
        # type: (str, int) -> SBValue
        type = self.valobj.GetType().GetBasicType(eBasicTypeUnsignedLong)
        data = SBData.CreateDataFromUInt64Array(self.endianness, self.pointer_size, [value])
        return self.valobj.CreateValueFromData(name, data, type)


def unwrap_unique_or_non_null(unique_or_nonnull):
    # BACKCOMPAT: rust 1.32
    # https://github.com/rust-lang/rust/commit/7a0911528058e87d22ea305695f4047572c5e067
    ptr = unique_or_nonnull.GetChildMemberWithName("pointer")
    return ptr if ptr.TypeIsPointerType() else ptr.GetChildAtIndex(0)


class DefaultSynthteticProvider:
    def __init__(self, valobj, dict):
        # type: (SBValue, dict) -> DefaultSynthteticProvider
        # logger = Logger.Logger()
        # logger >> "Default synthetic provider for " + str(valobj.GetName())
        self.valobj = valobj

    def num_children(self):
        # type: () -> int
        return self.valobj.GetNumChildren()

    def get_child_index(self, name):
        # type: (str) -> int
        return self.valobj.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        return self.valobj.GetChildAtIndex(index)

    def update(self):
        # type: () -> None
        pass

    def has_children(self):
        # type: () -> bool
        return self.valobj.MightHaveChildren()


class EmptySyntheticProvider:
    def __init__(self, valobj, dict):
        # type: (SBValue, dict) -> EmptySyntheticProvider
        # logger = Logger.Logger()
        # logger >> "[EmptySyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj

    def num_children(self):
        # type: () -> int
        return 0

    def get_child_index(self, name):
        # type: (str) -> int
        return None

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        return None

    def update(self):
        # type: () -> None
        pass

    def has_children(self):
        # type: () -> bool
        return False


def SizeSummaryProvider(valobj, dict):
    # type: (SBValue, dict) -> str
    return 'size=' + str(valobj.GetNumChildren())


def vec_to_string(vec):
    length = vec.GetNumChildren()
    chars = [vec.GetChildAtIndex(i).GetValueAsUnsigned() for i in range(length)]
    return bytes(chars).decode(errors='replace') if PY3 else "".join(chr(char) for char in chars)


def StdStringSummaryProvider(valobj, dict):
    # type: (SBValue, dict) -> str
    # logger = Logger.Logger()
    # logger >> "[StdStringSummaryProvider] for " + str(valobj.GetName())
    vec = valobj.GetChildAtIndex(0)
    return '"%s"' % vec_to_string(vec)


def StdOsStringSummaryProvider(valobj, dict):
    # type: (SBValue, dict) -> str
    # logger = Logger.Logger()
    # logger >> "[StdOsStringSummaryProvider] for " + str(valobj.GetName())
    buf = valobj.GetChildAtIndex(0).GetChildAtIndex(0)
    is_windows = "Wtf8Buf" in buf.type.name
    vec = buf.GetChildAtIndex(0) if is_windows else buf
    return '"%s"' % vec_to_string(vec)


def StdStrSummaryProvider(valobj, dict):
    # type: (SBValue, dict) -> str
    # logger = Logger.Logger()
    # logger >> "[StdStrSummaryProvider] for " + str(valobj.GetName())

    length = valobj.GetChildMemberWithName("length").GetValueAsUnsigned()
    if length == 0:
        return '""'

    data_ptr = valobj.GetChildMemberWithName("data_ptr")

    start = data_ptr.GetValueAsUnsigned()
    error = SBError()
    process = data_ptr.GetProcess()
    data = process.ReadMemory(start, length, error)
    data = data.decode(encoding='UTF-8') if PY3 else data
    return '"%s"' % data


class StructSyntheticProvider:
    """Pretty-printer for structs and struct enum variants"""

    def __init__(self, valobj, dict, is_variant=False):
        # type: (SBValue, dict, bool) -> StructSyntheticProvider
        # logger = Logger.Logger()
        self.valobj = valobj
        self.is_variant = is_variant
        self.type = valobj.GetType()
        self.fields = {}

        if is_variant:
            self.fields_count = self.type.GetNumberOfFields() - 1
            real_fields = self.type.fields[1:]
        else:
            self.fields_count = self.type.GetNumberOfFields()
            real_fields = self.type.fields

        for number, field in enumerate(real_fields):
            self.fields[field.name] = number

    def num_children(self):
        # type: () -> int
        return self.fields_count

    def get_child_index(self, name):
        # type: (str) -> int
        return self.fields.get(name, -1)

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        if self.is_variant:
            field = self.type.GetFieldAtIndex(index + 1)
        else:
            field = self.type.GetFieldAtIndex(index)
        return self.valobj.GetChildMemberWithName(field.name)

    def update(self):
        # type: () -> None
        pass

    def has_children(self):
        # type: () -> bool
        return True


class TupleSyntheticProvider:
    """Pretty-printer for tuples and tuple enum variants"""

    def __init__(self, valobj, dict, is_variant=False):
        # type: (SBValue, dict, bool) -> TupleSyntheticProvider
        # logger = Logger.Logger()
        self.valobj = valobj
        self.is_variant = is_variant
        self.type = valobj.GetType()

        if is_variant:
            self.size = self.type.GetNumberOfFields() - 1
        else:
            self.size = self.type.GetNumberOfFields()

    def num_children(self):
        # type: () -> int
        return self.size

    def get_child_index(self, name):
        # type: (str) -> int
        if name.isdigit():
            return int(name)
        else:
            return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        if self.is_variant:
            field = self.type.GetFieldAtIndex(index + 1)
        else:
            field = self.type.GetFieldAtIndex(index)
        element = self.valobj.GetChildMemberWithName(field.name)
        return self.valobj.CreateValueFromData(str(index), element.GetData(), element.GetType())

    def update(self):
        # type: () -> None
        pass

    def has_children(self):
        # type: () -> bool
        return True


class StdVecSyntheticProvider:
    """Pretty-printer for alloc::vec::Vec<T>

    struct Vec<T> { buf: RawVec<T>, len: usize }
    struct RawVec<T> { ptr: Unique<T>, cap: usize, ... }
    rust 1.31.1: struct Unique<T: ?Sized> { pointer: NonZero<*const T>, ... }
    rust 1.33.0: struct Unique<T: ?Sized> { pointer: *const T, ... }
    struct NonZero<T>(T)
    """

    def __init__(self, valobj, dict):
        # type: (SBValue, dict) -> StdVecSyntheticProvider
        # logger = Logger.Logger()
        # logger >> "[StdVecSyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj
        self.update()

    def num_children(self):
        # type: () -> int
        return self.length

    def get_child_index(self, name):
        # type: (str) -> int
        index = name.lstrip('[').rstrip(']')
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + index * self.element_type_size
        element = self.data_ptr.CreateValueFromAddress("[%s]" % index, address, self.element_type)
        return element

    def update(self):
        # type: () -> None
        self.length = self.valobj.GetChildMemberWithName("len").GetValueAsUnsigned()
        self.buf = self.valobj.GetChildMemberWithName("buf")

        self.data_ptr = unwrap_unique_or_non_null(self.buf.GetChildMemberWithName("ptr"))

        self.element_type = self.data_ptr.GetType().GetPointeeType()
        self.element_type_size = self.element_type.GetByteSize()

    def has_children(self):
        # type: () -> bool
        return True


class StdSliceSyntheticProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.update()

    def num_children(self):
        # type: () -> int
        return self.length

    def get_child_index(self, name):
        # type: (str) -> int
        index = name.lstrip('[').rstrip(']')
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + index * self.element_type_size
        element = self.data_ptr.CreateValueFromAddress("[%s]" % index, address, self.element_type)
        return element

    def update(self):
        # type: () -> None
        self.length = self.valobj.GetChildMemberWithName("length").GetValueAsUnsigned()
        self.data_ptr = self.valobj.GetChildMemberWithName("data_ptr")

        self.element_type = self.data_ptr.GetType().GetPointeeType()
        self.element_type_size = self.element_type.GetByteSize()

    def has_children(self):
        # type: () -> bool
        return True


class StdVecDequeSyntheticProvider:
    """Pretty-printer for alloc::collections::vec_deque::VecDeque<T>

    struct VecDeque<T> { tail: usize, head: usize, buf: RawVec<T> }
    """

    def __init__(self, valobj, dict):
        # type: (SBValue, dict) -> StdVecDequeSyntheticProvider
        # logger = Logger.Logger()
        # logger >> "[StdVecDequeSyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj
        self.update()

    def num_children(self):
        # type: () -> int
        return self.size

    def get_child_index(self, name):
        # type: (str) -> int
        index = name.lstrip('[').rstrip(']')
        if index.isdigit() and self.tail <= index and (self.tail + index) % self.cap < self.head:
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        start = self.data_ptr.GetValueAsUnsigned()
        address = start + ((index + self.tail) % self.cap) * self.element_type_size
        element = self.data_ptr.CreateValueFromAddress("[%s]" % index, address, self.element_type)
        return element

    def update(self):
        # type: () -> None
        self.head = self.valobj.GetChildMemberWithName("head").GetValueAsUnsigned()
        self.tail = self.valobj.GetChildMemberWithName("tail").GetValueAsUnsigned()
        self.buf = self.valobj.GetChildMemberWithName("buf")
        self.cap = self.buf.GetChildMemberWithName("cap").GetValueAsUnsigned()
        if self.head >= self.tail:
            self.size = self.head - self.tail
        else:
            self.size = self.cap + self.head - self.tail

        self.data_ptr = unwrap_unique_or_non_null(self.buf.GetChildMemberWithName("ptr"))

        self.element_type = self.data_ptr.GetType().GetPointeeType()
        self.element_type_size = self.element_type.GetByteSize()

    def has_children(self):
        # type: () -> bool
        return True


# BACKCOMPAT: rust 1.35
class StdOldHashMapSyntheticProvider:
    """Pretty-printer for std::collections::hash::map::HashMap<K, V, S>

    struct HashMap<K, V, S> {..., table: RawTable<K, V>, ... }
    struct RawTable<K, V> { capacity_mask: usize, size: usize, hashes: TaggedHashUintPtr, ... }
    """

    def __init__(self, valobj, dict, show_values=True):
        # type: (SBValue, dict, bool) -> StdOldHashMapSyntheticProvider
        self.valobj = valobj
        self.show_values = show_values
        self.update()

    def num_children(self):
        # type: () -> int
        return self.size

    def get_child_index(self, name):
        # type: (str) -> int
        index = name.lstrip('[').rstrip(']')
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        # logger = Logger.Logger()
        start = self.data_ptr.GetValueAsUnsigned() & ~1

        # See `libstd/collections/hash/table.rs:raw_bucket_at
        hashes = self.hash_uint_size * self.capacity
        align = self.pair_type_size
        # See `libcore/alloc.rs:padding_needed_for`
        len_rounded_up = (((((hashes + align) % self.modulo - 1) % self.modulo) & ~(
                (align - 1) % self.modulo)) % self.modulo - hashes) % self.modulo
        # len_rounded_up = ((hashes + align - 1) & ~(align - 1)) - hashes

        pairs_offset = hashes + len_rounded_up
        pairs_start = start + pairs_offset

        table_index = self.valid_indices[index]
        idx = table_index & self.capacity_mask
        address = pairs_start + idx * self.pair_type_size
        element = self.data_ptr.CreateValueFromAddress("[%s]" % index, address, self.pair_type)
        if self.show_values:
            return element
        else:
            key = element.GetChildAtIndex(0)
            return self.valobj.CreateValueFromData("[%s]" % index, key.GetData(), key.GetType())

    def update(self):
        # type: () -> None
        # logger = Logger.Logger()

        self.table = self.valobj.GetChildMemberWithName("table")  # type: SBValue
        self.size = self.table.GetChildMemberWithName("size").GetValueAsUnsigned()
        self.hashes = self.table.GetChildMemberWithName("hashes")
        self.hash_uint_type = self.hashes.GetType()
        self.hash_uint_size = self.hashes.GetType().GetByteSize()
        self.modulo = 2 ** self.hash_uint_size
        self.data_ptr = self.hashes.GetChildAtIndex(0).GetChildAtIndex(0)

        self.capacity_mask = self.table.GetChildMemberWithName("capacity_mask").GetValueAsUnsigned()
        self.capacity = (self.capacity_mask + 1) % self.modulo

        marker = self.table.GetChildMemberWithName("marker").GetType()  # type: SBType
        self.pair_type = marker.template_args[0]
        self.pair_type_size = self.pair_type.GetByteSize()

        self.valid_indices = []
        for idx in range(self.capacity):
            address = self.data_ptr.GetValueAsUnsigned() + idx * self.hash_uint_size
            hash_uint = self.data_ptr.CreateValueFromAddress("[%s]" % idx, address,
                                                             self.hash_uint_type)
            hash_ptr = hash_uint.GetChildAtIndex(0).GetChildAtIndex(0)
            if hash_ptr.GetValueAsUnsigned() != 0:
                self.valid_indices.append(idx)

        # logger >> "Valid indices: {}".format(str(self.valid_indices))

    def has_children(self):
        # type: () -> bool
        return True


class StdHashMapSyntheticProvider:
    """Pretty-printer for hashbrown's HashMap"""

    def __init__(self, valobj, dict, show_values=True):
        # type: (SBValue, dict, bool) -> StdHashMapSyntheticProvider
        self.valobj = valobj
        self.show_values = show_values
        self.update()

    def num_children(self):
        # type: () -> int
        return self.size

    def get_child_index(self, name):
        # type: (str) -> int
        index = name.lstrip('[').rstrip(']')
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        pairs_start = self.data_ptr.GetValueAsUnsigned()
        idx = self.valid_indices[index]
        if self.new_layout:
            idx = -(idx + 1)
        address = pairs_start + idx * self.pair_type_size
        element = self.data_ptr.CreateValueFromAddress("[%s]" % index, address, self.pair_type)
        if self.show_values:
            return element
        else:
            key = element.GetChildAtIndex(0)
            return self.valobj.CreateValueFromData("[%s]" % index, key.GetData(), key.GetType())

    def update(self):
        # type: () -> None
        table = self.table()
        capacity = table.GetChildMemberWithName("bucket_mask").GetValueAsUnsigned() + 1
        ctrl = table.GetChildMemberWithName("ctrl").GetChildAtIndex(0)

        self.size = table.GetChildMemberWithName("items").GetValueAsUnsigned()
        self.pair_type = table.type.template_args[0].GetTypedefedType()
        self.pair_type_size = self.pair_type.GetByteSize()

        self.new_layout = not table.GetChildMemberWithName("data").IsValid()
        if self.new_layout:
            self.data_ptr = ctrl.Cast(self.pair_type.GetPointerType())
        else:
            self.data_ptr = table.GetChildMemberWithName("data").GetChildAtIndex(0)

        u8_type = self.valobj.GetTarget().GetBasicType(eBasicTypeUnsignedChar)
        u8_type_size = self.valobj.GetTarget().GetBasicType(eBasicTypeUnsignedChar).GetByteSize()

        self.valid_indices = []
        for idx in range(capacity):
            address = ctrl.GetValueAsUnsigned() + idx * u8_type_size
            value = ctrl.CreateValueFromAddress("ctrl[%s]" % idx, address,
                                                u8_type).GetValueAsUnsigned()
            is_present = value & 128 == 0
            if is_present:
                self.valid_indices.append(idx)

    def table(self):
        # type: () -> SBValue
        if self.show_values:
            hashbrown_hashmap = self.valobj.GetChildMemberWithName("base")
        else:
            # BACKCOMPAT: rust 1.47
            # HashSet wraps either std HashMap or hashbrown::HashSet, which both
            # wrap hashbrown::HashMap, so either way we "unwrap" twice.
            hashbrown_hashmap = self.valobj.GetChildAtIndex(0).GetChildAtIndex(0)
        return hashbrown_hashmap.GetChildMemberWithName("table")

    def has_children(self):
        # type: () -> bool
        return True


def StdRcSummaryProvider(valobj, dict):
    # type: (SBValue, dict) -> str
    strong = valobj.GetChildMemberWithName("strong").GetValueAsUnsigned()
    weak = valobj.GetChildMemberWithName("weak").GetValueAsUnsigned()
    return "strong={}, weak={}".format(strong, weak)


class StdRcSyntheticProvider:
    """Pretty-printer for alloc::rc::Rc<T> and alloc::sync::Arc<T>

    struct Rc<T> { ptr: NonNull<RcBox<T>>, ... }
    rust 1.31.1: struct NonNull<T> { pointer: NonZero<*const T> }
    rust 1.33.0: struct NonNull<T> { pointer: *const T }
    struct NonZero<T>(T)
    struct RcBox<T> { strong: Cell<usize>, weak: Cell<usize>, value: T }
    struct Cell<T> { value: UnsafeCell<T> }
    struct UnsafeCell<T> { value: T }

    struct Arc<T> { ptr: NonNull<ArcInner<T>>, ... }
    struct ArcInner<T> { strong: atomic::AtomicUsize, weak: atomic::AtomicUsize, data: T }
    struct AtomicUsize { v: UnsafeCell<usize> }
    """

    def __init__(self, valobj, dict, is_atomic=False):
        # type: (SBValue, dict, bool) -> StdRcSyntheticProvider
        self.valobj = valobj

        self.ptr = unwrap_unique_or_non_null(self.valobj.GetChildMemberWithName("ptr"))

        self.value = self.ptr.GetChildMemberWithName("data" if is_atomic else "value")

        self.strong = self.ptr.GetChildMemberWithName("strong").GetChildAtIndex(
            0).GetChildMemberWithName("value")
        self.weak = self.ptr.GetChildMemberWithName("weak").GetChildAtIndex(
            0).GetChildMemberWithName("value")

        self.value_builder = ValueBuilder(valobj)

        self.update()

    def num_children(self):
        # type: () -> int
        # Actually there are 3 children, but only the `value` should be shown as a child
        return 1

    def get_child_index(self, name):
        # type: (str) -> int
        if name == "value":
            return 0
        if name == "strong":
            return 1
        if name == "weak":
            return 2
        return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        if index == 0:
            return self.value
        if index == 1:
            return self.value_builder.from_uint("strong", self.strong_count)
        if index == 2:
            return self.value_builder.from_uint("weak", self.weak_count)

        return None

    def update(self):
        # type: () -> None
        self.strong_count = self.strong.GetValueAsUnsigned()
        self.weak_count = self.weak.GetValueAsUnsigned() - 1

    def has_children(self):
        # type: () -> bool
        return True


class StdCellSyntheticProvider:
    """Pretty-printer for std::cell::Cell"""

    def __init__(self, valobj, dict):
        # type: (SBValue, dict) -> StdCellSyntheticProvider
        self.valobj = valobj
        self.value = valobj.GetChildMemberWithName("value").GetChildAtIndex(0)

    def num_children(self):
        # type: () -> int
        return 1

    def get_child_index(self, name):
        # type: (str) -> int
        if name == "value":
            return 0
        return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        if index == 0:
            return self.value
        return None

    def update(self):
        # type: () -> None
        pass

    def has_children(self):
        # type: () -> bool
        return True


def StdRefSummaryProvider(valobj, dict):
    # type: (SBValue, dict) -> str
    borrow = valobj.GetChildMemberWithName("borrow").GetValueAsSigned()
    return "borrow={}".format(borrow) if borrow >= 0 else "borrow_mut={}".format(-borrow)


class StdRefSyntheticProvider:
    """Pretty-printer for std::cell::Ref, std::cell::RefMut, and std::cell::RefCell"""

    def __init__(self, valobj, dict, is_cell=False):
        # type: (SBValue, dict, bool) -> StdRefSyntheticProvider
        self.valobj = valobj

        borrow = valobj.GetChildMemberWithName("borrow")
        value = valobj.GetChildMemberWithName("value")
        if is_cell:
            self.borrow = borrow.GetChildMemberWithName("value").GetChildMemberWithName("value")
            self.value = value.GetChildMemberWithName("value")
        else:
            self.borrow = borrow.GetChildMemberWithName("borrow").GetChildMemberWithName(
                "value").GetChildMemberWithName("value")
            self.value = value.Dereference()

        self.value_builder = ValueBuilder(valobj)

        self.update()

    def num_children(self):
        # type: () -> int
        # Actually there are 2 children, but only the `value` should be shown as a child
        return 1

    def get_child_index(self, name):
        if name == "value":
            return 0
        if name == "borrow":
            return 1
        return -1

    def get_child_at_index(self, index):
        # type: (int) -> SBValue
        if index == 0:
            return self.value
        if index == 1:
            return self.value_builder.from_int("borrow", self.borrow_count)
        return None

    def update(self):
        # type: () -> None
        self.borrow_count = self.borrow.GetValueAsSigned()

    def has_children(self):
        # type: () -> bool
        return True
