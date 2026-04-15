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

use melior::Context;
use melior::ir::Location;
use rustc_middle::ty::TyCtxt;
use rustc_span::{Pos, Span};

/// Convert a rustc `Span` to an MLIR `Location` (FileLineColLoc).
///
/// For macro-expanded spans, resolves to the outermost call site in user
/// source via `source_callsite()`. Falls back to `Location::unknown` for
/// dummy spans (compiler-synthesized code with no real source position).
///
/// MLIR line/column are both 1-based; rustc's `CharPos` column is 0-based,
/// so we add 1 when constructing the location.
pub fn span_to_location<'ctx>(
    context: &'ctx Context,
    tcx: TyCtxt<'_>,
    span: Span,
) -> Location<'ctx> {
    if span.is_dummy() {
        return Location::unknown(context);
    }

    // For macro-expanded spans, use the outermost call site in user code.
    let span = span.source_callsite();

    if span.is_dummy() {
        return Location::unknown(context);
    }

    let source_map = tcx.sess.source_map();
    let loc = source_map.lookup_char_pos(span.lo());
    let filename = format!("{}", loc.file.name.prefer_remapped_unconditionally());

    // MLIR uses 1-based line and column numbers; rustc CharPos column is 0-based.
    Location::new(context, &filename, loc.line, loc.col.to_usize() + 1)
}
