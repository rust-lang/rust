from typing import List
import re


class RustType(object):
    OTHER = "Other"
    STRUCT = "Struct"
    TUPLE = "Tuple"
    CSTYLE_VARIANT = "CStyleVariant"
    TUPLE_VARIANT = "TupleVariant"
    STRUCT_VARIANT = "StructVariant"
    ENUM = "Enum"
    EMPTY = "Empty"
    SINGLETON_ENUM = "SingletonEnum"
    REGULAR_ENUM = "RegularEnum"
    COMPRESSED_ENUM = "CompressedEnum"
    REGULAR_UNION = "RegularUnion"

    STD_STRING = "StdString"
    STD_OS_STRING = "StdOsString"
    STD_STR = "StdStr"
    STD_SLICE = "StdSlice"
    STD_VEC = "StdVec"
    STD_VEC_DEQUE = "StdVecDeque"
    STD_BTREE_SET = "StdBTreeSet"
    STD_BTREE_MAP = "StdBTreeMap"
    STD_HASH_MAP = "StdHashMap"
    STD_HASH_SET = "StdHashSet"
    STD_RC = "StdRc"
    STD_ARC = "StdArc"
    STD_CELL = "StdCell"
    STD_REF = "StdRef"
    STD_REF_MUT = "StdRefMut"
    STD_REF_CELL = "StdRefCell"
    STD_NONZERO_NUMBER = "StdNonZeroNumber"
    STD_PATH = "StdPath"
    STD_PATHBUF = "StdPathBuf"


STD_STRING_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)String$")
STD_STR_REGEX = re.compile(r"^&(mut )?str$")
STD_SLICE_REGEX = re.compile(r"^&(mut )?\[.+\]$")
STD_OS_STRING_REGEX = re.compile(r"^(std::ffi::([a-z_]+::)+)OsString$")
STD_VEC_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)Vec<.+>$")
STD_VEC_DEQUE_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)VecDeque<.+>$")
STD_BTREE_SET_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)BTreeSet<.+>$")
STD_BTREE_MAP_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)BTreeMap<.+>$")
STD_HASH_MAP_REGEX = re.compile(r"^(std::collections::([a-z_]+::)+)HashMap<.+>$")
STD_HASH_SET_REGEX = re.compile(r"^(std::collections::([a-z_]+::)+)HashSet<.+>$")
STD_RC_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)Rc<.+>$")
STD_ARC_REGEX = re.compile(r"^(alloc::([a-z_]+::)+)Arc<.+>$")
STD_CELL_REGEX = re.compile(r"^(core::([a-z_]+::)+)Cell<.+>$")
STD_REF_REGEX = re.compile(r"^(core::([a-z_]+::)+)Ref<.+>$")
STD_REF_MUT_REGEX = re.compile(r"^(core::([a-z_]+::)+)RefMut<.+>$")
STD_REF_CELL_REGEX = re.compile(r"^(core::([a-z_]+::)+)RefCell<.+>$")
STD_NONZERO_NUMBER_REGEX = re.compile(r"^(core::([a-z_]+::)+)NonZero<.+>$")
STD_PATHBUF_REGEX = re.compile(r"^(std::([a-z_]+::)+)PathBuf$")
STD_PATH_REGEX = re.compile(r"^&(mut )?(std::([a-z_]+::)+)Path$")

TUPLE_ITEM_REGEX = re.compile(r"__\d+$")

ENCODED_ENUM_PREFIX = "RUST$ENCODED$ENUM$"
ENUM_DISR_FIELD_NAME = "<<variant>>"
ENUM_LLDB_ENCODED_VARIANTS = "$variants$"

STD_TYPE_TO_REGEX = {
    RustType.STD_STRING: STD_STRING_REGEX,
    RustType.STD_OS_STRING: STD_OS_STRING_REGEX,
    RustType.STD_STR: STD_STR_REGEX,
    RustType.STD_SLICE: STD_SLICE_REGEX,
    RustType.STD_VEC: STD_VEC_REGEX,
    RustType.STD_VEC_DEQUE: STD_VEC_DEQUE_REGEX,
    RustType.STD_HASH_MAP: STD_HASH_MAP_REGEX,
    RustType.STD_HASH_SET: STD_HASH_SET_REGEX,
    RustType.STD_BTREE_SET: STD_BTREE_SET_REGEX,
    RustType.STD_BTREE_MAP: STD_BTREE_MAP_REGEX,
    RustType.STD_RC: STD_RC_REGEX,
    RustType.STD_ARC: STD_ARC_REGEX,
    RustType.STD_REF: STD_REF_REGEX,
    RustType.STD_REF_MUT: STD_REF_MUT_REGEX,
    RustType.STD_REF_CELL: STD_REF_CELL_REGEX,
    RustType.STD_CELL: STD_CELL_REGEX,
    RustType.STD_NONZERO_NUMBER: STD_NONZERO_NUMBER_REGEX,
    RustType.STD_PATHBUF: STD_PATHBUF_REGEX,
    RustType.STD_PATH: STD_PATH_REGEX,
}


def is_tuple_fields(fields: List) -> bool:
    return all(TUPLE_ITEM_REGEX.match(str(field.name)) for field in fields)


def classify_struct(name: str, fields: List) -> str:
    if len(fields) == 0:
        return RustType.EMPTY

    for ty, regex in STD_TYPE_TO_REGEX.items():
        if regex.match(name):
            return ty

    # <<variant>> is emitted by GDB while LLDB(18.1+) emits "$variants$"
    if (
        fields[0].name == ENUM_DISR_FIELD_NAME
        or fields[0].name == ENUM_LLDB_ENCODED_VARIANTS
    ):
        return RustType.ENUM

    if is_tuple_fields(fields):
        return RustType.TUPLE

    return RustType.STRUCT


def classify_union(fields: List) -> str:
    if len(fields) == 0:
        return RustType.EMPTY

    first_variant_name = fields[0].name
    if first_variant_name is None:
        if len(fields) == 1:
            return RustType.SINGLETON_ENUM
        else:
            return RustType.REGULAR_ENUM
    elif first_variant_name.startswith(ENCODED_ENUM_PREFIX):
        assert len(fields) == 1
        return RustType.COMPRESSED_ENUM
    else:
        return RustType.REGULAR_UNION
