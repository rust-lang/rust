#![feature(rustc_private)]

use clippy_utils::paths::{PathLookup, PathNS};
use clippy_utils::{macro_path, sym, type_path, value_path};

static OPTION: PathLookup = type_path!(core::option::Option);
//~^ unnecessary_def_path
static SOME: PathLookup = type_path!(core::option::Option::Some);
//~^ unnecessary_def_path

static RESULT: PathLookup = type_path!(core::result::Result);
//~^ unnecessary_def_path
static RESULT_VIA_STD: PathLookup = type_path!(std::result::Result);
//~^ unnecessary_def_path

static VEC_NEW: PathLookup = value_path!(alloc::vec::Vec::new);
//~^ unnecessary_def_path

static VEC_MACRO: PathLookup = macro_path!(std::vec);
//~^ unnecessary_def_path
