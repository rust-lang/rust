import gdb
import re
import sys
import debugger_pretty_printers_common as rustpp

# We want a version of `range` which doesn't allocate an intermediate list,
# specifically it should use a lazy iterator. In Python 2 this was `xrange`, but
# if we're running with Python 3 then we need to use `range` instead.
if sys.version_info[0] >= 3:
    xrange = range

rust_enabled = 'set language rust' in gdb.execute('complete set language ru', to_string = True)

# The btree pretty-printers fail in a confusing way unless
# https://sourceware.org/bugzilla/show_bug.cgi?id=21763 is fixed.
# This fix went in 8.1, so check for that.
# See https://github.com/rust-lang/rust/issues/56730
gdb_81 = False
_match = re.search('([0-9]+)\\.([0-9]+)', gdb.VERSION)
if _match:
    if int(_match.group(1)) > 8 or (int(_match.group(1)) == 8 and int(_match.group(2)) >= 1):
        gdb_81 = True

#===============================================================================
# GDB Pretty Printing Module for Rust
#===============================================================================

class GdbType(rustpp.Type):

    def __init__(self, ty):
        super(GdbType, self).__init__()
        self.ty = ty
        self.fields = None

    def get_unqualified_type_name(self):
        tag = self.ty.tag

        if tag is None:
            return tag

        return rustpp.extract_type_name(tag).replace("&'static ", "&")

    def get_dwarf_type_kind(self):
        if self.ty.code == gdb.TYPE_CODE_STRUCT:
            return rustpp.DWARF_TYPE_CODE_STRUCT

        if self.ty.code == gdb.TYPE_CODE_UNION:
            return rustpp.DWARF_TYPE_CODE_UNION

        if self.ty.code == gdb.TYPE_CODE_PTR:
            return rustpp.DWARF_TYPE_CODE_PTR

        if self.ty.code == gdb.TYPE_CODE_ENUM:
            return rustpp.DWARF_TYPE_CODE_ENUM

    def get_fields(self):
        assert ((self.get_dwarf_type_kind() == rustpp.DWARF_TYPE_CODE_STRUCT) or
                (self.get_dwarf_type_kind() == rustpp.DWARF_TYPE_CODE_UNION))
        if self.fields is None:
            self.fields = list(self.ty.fields())
        return self.fields

    def get_wrapped_value(self):
        return self.ty


class GdbValue(rustpp.Value):
    def __init__(self, gdb_val):
        super(GdbValue, self).__init__(GdbType(gdb_val.type))
        self.gdb_val = gdb_val
        self.children = {}

    def get_child_at_index(self, index):
        child = self.children.get(index)
        if child is None:
            gdb_field = get_field_at_index(self.gdb_val, index)
            child = GdbValue(self.gdb_val[gdb_field])
            self.children[index] = child
        return child

    def as_integer(self):
        if self.gdb_val.type.code == gdb.TYPE_CODE_PTR:
            as_str = rustpp.compat_str(self.gdb_val).split()[0]
            return int(as_str, 0)
        return int(self.gdb_val)

    def get_wrapped_value(self):
        return self.gdb_val


def register_printers(objfile):
    """Registers Rust pretty printers for the given objfile"""
    objfile.pretty_printers.append(rust_pretty_printer_lookup_function)


def rust_pretty_printer_lookup_function(gdb_val):
    """
    Returns the correct Rust pretty printer for the given value
    if there is one
    """

    val = GdbValue(gdb_val)
    type_kind = val.type.get_type_kind()

    if type_kind == rustpp.TYPE_KIND_SLICE:
        return RustSlicePrinter(val)

    if type_kind == rustpp.TYPE_KIND_STD_VEC:
        return RustStdVecPrinter(val)

    if type_kind == rustpp.TYPE_KIND_STD_VECDEQUE:
        return RustStdVecDequePrinter(val)

    if type_kind == rustpp.TYPE_KIND_STD_BTREESET and gdb_81:
        return RustStdBTreeSetPrinter(val)

    if type_kind == rustpp.TYPE_KIND_STD_BTREEMAP and gdb_81:
        return RustStdBTreeMapPrinter(val)

    if type_kind == rustpp.TYPE_KIND_STD_STRING:
        return RustStdStringPrinter(val)

    if type_kind == rustpp.TYPE_KIND_OS_STRING:
        return RustOsStringPrinter(val)

    # Checks after this point should only be for "compiler" types --
    # things that gdb's Rust language support knows about.
    if rust_enabled:
        return None

    if type_kind == rustpp.TYPE_KIND_EMPTY:
        return RustEmptyPrinter(val)

    if type_kind == rustpp.TYPE_KIND_REGULAR_STRUCT:
        return RustStructPrinter(val,
                                 omit_first_field = False,
                                 omit_type_name = False,
                                 is_tuple_like = False)

    if type_kind == rustpp.TYPE_KIND_STRUCT_VARIANT:
        return RustStructPrinter(val,
                                 omit_first_field = True,
                                 omit_type_name = False,
                                 is_tuple_like = False)

    if type_kind == rustpp.TYPE_KIND_STR_SLICE:
        return RustStringSlicePrinter(val)

    if type_kind == rustpp.TYPE_KIND_TUPLE:
        return RustStructPrinter(val,
                                 omit_first_field = False,
                                 omit_type_name = True,
                                 is_tuple_like = True)

    if type_kind == rustpp.TYPE_KIND_TUPLE_STRUCT:
        return RustStructPrinter(val,
                                 omit_first_field = False,
                                 omit_type_name = False,
                                 is_tuple_like = True)

    if type_kind == rustpp.TYPE_KIND_CSTYLE_VARIANT:
        return RustCStyleVariantPrinter(val.get_child_at_index(0))

    if type_kind == rustpp.TYPE_KIND_TUPLE_VARIANT:
        return RustStructPrinter(val,
                                 omit_first_field = True,
                                 omit_type_name = False,
                                 is_tuple_like = True)

    if type_kind == rustpp.TYPE_KIND_SINGLETON_ENUM:
        variant = get_field_at_index(gdb_val, 0)
        return rust_pretty_printer_lookup_function(gdb_val[variant])

    if type_kind == rustpp.TYPE_KIND_REGULAR_ENUM:
        # This is a regular enum, extract the discriminant
        discriminant_val = rustpp.get_discriminant_value_as_integer(val)
        variant = get_field_at_index(gdb_val, discriminant_val)
        return rust_pretty_printer_lookup_function(gdb_val[variant])

    if type_kind == rustpp.TYPE_KIND_COMPRESSED_ENUM:
        encoded_enum_info = rustpp.EncodedEnumInfo(val)
        if encoded_enum_info.is_null_variant():
            return IdentityPrinter(encoded_enum_info.get_null_variant_name())

        non_null_val = encoded_enum_info.get_non_null_variant_val()
        return rust_pretty_printer_lookup_function(non_null_val.get_wrapped_value())

    # No pretty printer has been found
    return None


#=------------------------------------------------------------------------------
# Pretty Printer Classes
#=------------------------------------------------------------------------------
class RustEmptyPrinter(object):
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        return self.__val.type.get_unqualified_type_name()


class RustStructPrinter(object):
    def __init__(self, val, omit_first_field, omit_type_name, is_tuple_like):
        self.__val = val
        self.__omit_first_field = omit_first_field
        self.__omit_type_name = omit_type_name
        self.__is_tuple_like = is_tuple_like

    def to_string(self):
        if self.__omit_type_name:
            return None
        return self.__val.type.get_unqualified_type_name()

    def children(self):
        cs = []
        wrapped_value = self.__val.get_wrapped_value()

        for number, field in enumerate(self.__val.type.get_fields()):
            field_value = wrapped_value[field.name]
            if self.__is_tuple_like:
                cs.append((str(number), field_value))
            else:
                cs.append((field.name, field_value))

        if self.__omit_first_field:
            cs = cs[1:]

        return cs

    def display_hint(self):
        if self.__is_tuple_like:
            return "array"
        else:
            return ""


class RustSlicePrinter(object):
    def __init__(self, val):
        self.__val = val

    @staticmethod
    def display_hint():
        return "array"

    def to_string(self):
        (length, data_ptr) = rustpp.extract_length_and_ptr_from_slice(self.__val)
        return (self.__val.type.get_unqualified_type_name() +
                ("(len: %i)" % length))

    def children(self):
        (length, data_ptr) = rustpp.extract_length_and_ptr_from_slice(self.__val)
        assert data_ptr.type.get_dwarf_type_kind() == rustpp.DWARF_TYPE_CODE_PTR
        raw_ptr = data_ptr.get_wrapped_value()

        for index in xrange(0, length):
            yield (str(index), (raw_ptr + index).dereference())


class RustStringSlicePrinter(object):
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        (length, data_ptr) = rustpp.extract_length_and_ptr_from_slice(self.__val)
        raw_ptr = data_ptr.get_wrapped_value()
        return raw_ptr.lazy_string(encoding="utf-8", length=length)

    def display_hint(self):
        return "string"


class RustStdVecPrinter(object):
    def __init__(self, val):
        self.__val = val

    @staticmethod
    def display_hint():
        return "array"

    def to_string(self):
        (length, data_ptr, cap) = rustpp.extract_length_ptr_and_cap_from_std_vec(self.__val)
        return (self.__val.type.get_unqualified_type_name() +
                ("(len: %i, cap: %i)" % (length, cap)))

    def children(self):
        (length, data_ptr, cap) = rustpp.extract_length_ptr_and_cap_from_std_vec(self.__val)
        gdb_ptr = data_ptr.get_wrapped_value()
        for index in xrange(0, length):
            yield (str(index), (gdb_ptr + index).dereference())


class RustStdVecDequePrinter(object):
    def __init__(self, val):
        self.__val = val

    @staticmethod
    def display_hint():
        return "array"

    def to_string(self):
        (tail, head, data_ptr, cap) = \
            rustpp.extract_tail_head_ptr_and_cap_from_std_vecdeque(self.__val)
        if head >= tail:
            size = head - tail
        else:
            size = cap + head - tail
        return (self.__val.type.get_unqualified_type_name() +
                ("(len: %i, cap: %i)" % (size, cap)))

    def children(self):
        (tail, head, data_ptr, cap) = \
            rustpp.extract_tail_head_ptr_and_cap_from_std_vecdeque(self.__val)
        gdb_ptr = data_ptr.get_wrapped_value()
        if head >= tail:
            size = head - tail
        else:
            size = cap + head - tail
        for index in xrange(0, size):
            yield (str(index), (gdb_ptr + ((tail + index) % cap)).dereference())


# Yield each key (and optionally value) from a BoxedNode.
def children_of_node(boxed_node, height, want_values):
    node_ptr = boxed_node['ptr']['pointer']
    if height > 0:
        type_name = str(node_ptr.type.target()).replace('LeafNode', 'InternalNode')
        node_type = gdb.lookup_type(type_name)
        node_ptr = node_ptr.cast(node_type.pointer())
        leaf = node_ptr['data']
    else:
        leaf = node_ptr.dereference()
    keys = leaf['keys']
    if want_values:
        values = leaf['vals']
    length = int(leaf['len'])
    for i in xrange(0, length + 1):
        if height > 0:
            child_ptr = node_ptr['edges'][i]['value']['value']
            for child in children_of_node(child_ptr, height - 1, want_values):
                yield child
        if i < length:
            if want_values:
                yield (keys[i]['value']['value'], values[i]['value']['value'])
            else:
                yield keys[i]['value']['value']

class RustStdBTreeSetPrinter(object):
    def __init__(self, val):
        self.__val = val

    @staticmethod
    def display_hint():
        return "array"

    def to_string(self):
        return (self.__val.type.get_unqualified_type_name() +
                ("(len: %i)" % self.__val.get_wrapped_value()['map']['length']))

    def children(self):
        root = self.__val.get_wrapped_value()['map']['root']
        node_ptr = root['node']
        i = 0
        for child in children_of_node(node_ptr, root['height'], False):
            yield (str(i), child)
            i = i + 1


class RustStdBTreeMapPrinter(object):
    def __init__(self, val):
        self.__val = val

    @staticmethod
    def display_hint():
        return "map"

    def to_string(self):
        return (self.__val.type.get_unqualified_type_name() +
                ("(len: %i)" % self.__val.get_wrapped_value()['length']))

    def children(self):
        root = self.__val.get_wrapped_value()['root']
        node_ptr = root['node']
        i = 0
        for child in children_of_node(node_ptr, root['height'], True):
            yield (str(i), child[0])
            yield (str(i), child[1])
            i = i + 1


class RustStdStringPrinter(object):
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        vec = self.__val.get_child_at_index(0)
        (length, data_ptr, cap) = rustpp.extract_length_ptr_and_cap_from_std_vec(vec)
        return data_ptr.get_wrapped_value().lazy_string(encoding="utf-8",
                                                        length=length)

    def display_hint(self):
        return "string"


class RustOsStringPrinter(object):
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        buf = self.__val.get_child_at_index(0)
        vec = buf.get_child_at_index(0)
        if vec.type.get_unqualified_type_name() == "Wtf8Buf":
            vec = vec.get_child_at_index(0)

        (length, data_ptr, cap) = rustpp.extract_length_ptr_and_cap_from_std_vec(
            vec)
        return data_ptr.get_wrapped_value().lazy_string(length=length)

    def display_hint(self):
        return "string"

class RustCStyleVariantPrinter(object):
    def __init__(self, val):
        assert val.type.get_dwarf_type_kind() == rustpp.DWARF_TYPE_CODE_ENUM
        self.__val = val

    def to_string(self):
        return str(self.__val.get_wrapped_value())


class IdentityPrinter(object):
    def __init__(self, string):
        self.string = string

    def to_string(self):
        return self.string


def get_field_at_index(gdb_val, index):
    i = 0
    for field in gdb_val.type.fields():
        if i == index:
            return field
        i += 1
    return None
