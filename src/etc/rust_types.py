from enum import Enum
from typing import List
import re


class RustType(Enum):
    Other = 0
    Struct = 1
    Tuple = 2
    CStyleVariant = 3
    TupleVariant = 4
    StructVariant = 5
    Enum = 6
    Empty = 7
    SingletonEnum = 8
    RegularEnum = 9
    CompressedEnum = 10
    Union = 11
    Indirection = 12

    StdString = 13
    StdOsString = 14
    StdStr = 15
    StdSlice = 16
    StdVec = 17
    StdVecDeque = 18
    StdBTreeSet = 19
    StdBTreeMap = 20
    StdHashMap = 21
    StdHashSet = 22
    StdRc = 23
    StdArc = 24
    StdCell = 25
    StdRef = 26
    StdRefMut = 27
    StdRefCell = 28
    StdNonZeroNumber = 29
    StdPath = 30
    StdPathBuf = 31


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

ENCODED_ENUM_PREFIX = "RUST$ENCODED$ENUM$"
ENUM_DISR_FIELD_NAME = "<<variant>>"
ENUM_LLDB_ENCODED_VARIANTS = "$variants$"

STD_TYPE_TO_REGEX = {
    RustType.StdString: STD_STRING_REGEX,
    RustType.StdOsString: STD_OS_STRING_REGEX,
    RustType.StdStr: STD_STR_REGEX,
    RustType.StdSlice: STD_SLICE_REGEX,
    RustType.StdVec: STD_VEC_REGEX,
    RustType.StdVecDeque: STD_VEC_DEQUE_REGEX,
    RustType.StdHashMap: STD_HASH_MAP_REGEX,
    RustType.StdHashSet: STD_HASH_SET_REGEX,
    RustType.StdBTreeSet: STD_BTREE_SET_REGEX,
    RustType.StdBTreeMap: STD_BTREE_MAP_REGEX,
    RustType.StdRc: STD_RC_REGEX,
    RustType.StdArc: STD_ARC_REGEX,
    RustType.StdRef: STD_REF_REGEX,
    RustType.StdRefMut: STD_REF_MUT_REGEX,
    RustType.StdRefCell: STD_REF_CELL_REGEX,
    RustType.StdCell: STD_CELL_REGEX,
    RustType.StdNonZeroNumber: STD_NONZERO_NUMBER_REGEX,
    RustType.StdPath: STD_PATH_REGEX,
    RustType.StdPathBuf: STD_PATHBUF_REGEX,
}


def is_tuple_fields(fields: List) -> bool:
    for f in fields:
        name = str(f.name)
        if not name.startswith("__") or not name[2:].isdigit():
            return False

    return True


def classify_struct(name: str, fields: List) -> RustType:
    if len(fields) == 0:
        return RustType.Empty

    for ty, regex in STD_TYPE_TO_REGEX.items():
        if regex.match(name):
            return ty

    # <<variant>> is emitted by GDB while LLDB(18.1+) emits "$variants$"
    if (
        fields[0].name == ENUM_DISR_FIELD_NAME
        or fields[0].name == ENUM_LLDB_ENCODED_VARIANTS
    ):
        return RustType.Enum

    if is_tuple_fields(fields):
        return RustType.Tuple

    return RustType.Struct


def classify_union(fields: List) -> RustType:
    if len(fields) == 0:
        return RustType.Empty

    first_variant_name = fields[0].name
    if first_variant_name is None:
        if len(fields) == 1:
            return RustType.SingletonEnum
        else:
            return RustType.RegularEnum
    elif first_variant_name.startswith(ENCODED_ENUM_PREFIX):
        assert len(fields) == 1
        return RustType.CompressedEnum
    else:
        return RustType.RegularEnum
