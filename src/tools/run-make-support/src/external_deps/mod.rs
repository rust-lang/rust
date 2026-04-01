//! This module contains external tool dependencies that we assume are available in the environment,
//! such as `cc` or `python`.

pub mod c_build;
pub mod c_cxx_compiler;
pub mod cargo;
pub mod clang;
pub mod htmldocck;
pub mod llvm;
pub mod python;
pub mod rustc;
pub mod rustdoc;
