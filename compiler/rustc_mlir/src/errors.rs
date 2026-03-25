/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::fmt;

use rustc_errors::into_diag_arg_using_display;
use rustc_macros::Diagnostic;

#[derive(Debug, Diagnostic)]
pub enum Error {
    #[diag(mlir_invalid_type)]
    InvalidType { msg: String },

    #[diag(mlir_incompatible_types)]
    IncompatibleTypes { lhs: String, rhs: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidType { msg } => write!(f, "invalid type: {msg}"),
            Error::IncompatibleTypes { lhs, rhs } => {
                write!(f, "incompatible types: {lhs} vs {rhs}")
            }
        }
    }
}

into_diag_arg_using_display!(Error);

/// Result type for MLIR operations
pub type Result<T> = std::result::Result<T, Error>;
