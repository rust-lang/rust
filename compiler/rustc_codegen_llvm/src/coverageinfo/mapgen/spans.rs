use rustc_middle::mir::coverage::FunctionCoverageInfo;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, Pos, SourceFile, Span};
use tracing::debug;

use crate::coverageinfo::ffi;
use crate::coverageinfo::mapgen::LocalFileId;

/// Converts the span into its start line and column, and end line and column.
///
/// Line numbers and column numbers are 1-based. Unlike most column numbers emitted by
/// the compiler, these column numbers are denoted in **bytes**, because that's what
/// LLVM's `llvm-cov` tool expects to see in coverage maps.
///
/// Returns `None` if the conversion failed for some reason. This shouldn't happen,
/// but it's hard to rule out entirely (especially in the presence of complex macros
/// or other expansions), and if it does happen then skipping a span or function is
/// better than an ICE or `llvm-cov` failure that the user might have no way to avoid.
pub(crate) fn make_coverage_span(
    file_id: LocalFileId,
    source_map: &SourceMap,
    fn_cov_info: &FunctionCoverageInfo,
    file: &SourceFile,
    span: Span,
) -> Option<ffi::CoverageSpan> {
    let span = ensure_non_empty_span(source_map, fn_cov_info, span)?;

    let lo = span.lo();
    let hi = span.hi();

    // Column numbers need to be in bytes, so we can't use the more convenient
    // `SourceMap` methods for looking up file coordinates.
    let line_and_byte_column = |pos: BytePos| -> Option<(usize, usize)> {
        let rpos = file.relative_position(pos);
        let line_index = file.lookup_line(rpos)?;
        let line_start = file.lines()[line_index];
        // Line numbers and column numbers are 1-based, so add 1 to each.
        Some((line_index + 1, (rpos - line_start).to_usize() + 1))
    };

    let (mut start_line, start_col) = line_and_byte_column(lo)?;
    let (mut end_line, end_col) = line_and_byte_column(hi)?;

    // Apply an offset so that code in doctests has correct line numbers.
    // FIXME(#79417): Currently we have no way to offset doctest _columns_.
    start_line = source_map.doctest_offset_line(&file.name, start_line);
    end_line = source_map.doctest_offset_line(&file.name, end_line);

    check_coverage_span(ffi::CoverageSpan {
        file_id: file_id.as_u32(),
        start_line: start_line as u32,
        start_col: start_col as u32,
        end_line: end_line as u32,
        end_col: end_col as u32,
    })
}

fn ensure_non_empty_span(
    source_map: &SourceMap,
    fn_cov_info: &FunctionCoverageInfo,
    span: Span,
) -> Option<Span> {
    if !span.is_empty() {
        return Some(span);
    }

    let lo = span.lo();
    let hi = span.hi();

    // The span is empty, so try to expand it to cover an adjacent '{' or '}',
    // but only within the bounds of the body span.
    let try_next = hi < fn_cov_info.body_span.hi();
    let try_prev = fn_cov_info.body_span.lo() < lo;
    if !(try_next || try_prev) {
        return None;
    }

    source_map
        .span_to_source(span, |src, start, end| try {
            // Adjusting span endpoints by `BytePos(1)` is normally a bug,
            // but in this case we have specifically checked that the character
            // we're skipping over is one of two specific ASCII characters, so
            // adjusting by exactly 1 byte is correct.
            if try_next && src.as_bytes()[end] == b'{' {
                Some(span.with_hi(hi + BytePos(1)))
            } else if try_prev && src.as_bytes()[start - 1] == b'}' {
                Some(span.with_lo(lo - BytePos(1)))
            } else {
                None
            }
        })
        .ok()?
}

/// If `llvm-cov` sees a source region that is improperly ordered (end < start),
/// it will immediately exit with a fatal error. To prevent that from happening,
/// discard regions that are improperly ordered, or might be interpreted in a
/// way that makes them improperly ordered.
fn check_coverage_span(cov_span: ffi::CoverageSpan) -> Option<ffi::CoverageSpan> {
    let ffi::CoverageSpan { file_id: _, start_line, start_col, end_line, end_col } = cov_span;

    // Line/column coordinates are supposed to be 1-based. If we ever emit
    // coordinates of 0, `llvm-cov` might misinterpret them.
    let all_nonzero = [start_line, start_col, end_line, end_col].into_iter().all(|x| x != 0);
    // Coverage mappings use the high bit of `end_col` to indicate that a
    // region is actually a "gap" region, so make sure it's unset.
    let end_col_has_high_bit_unset = (end_col & (1 << 31)) == 0;
    // If a region is improperly ordered (end < start), `llvm-cov` will exit
    // with a fatal error, which is inconvenient for users and hard to debug.
    let is_ordered = (start_line, start_col) <= (end_line, end_col);

    if all_nonzero && end_col_has_high_bit_unset && is_ordered {
        Some(cov_span)
    } else {
        debug!(
            ?cov_span,
            ?all_nonzero,
            ?end_col_has_high_bit_unset,
            ?is_ordered,
            "Skipping source region that would be misinterpreted or rejected by LLVM"
        );
        // If this happens in a debug build, ICE to make it easier to notice.
        debug_assert!(false, "Improper source region: {cov_span:?}");
        None
    }
}
