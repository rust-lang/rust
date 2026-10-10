use std::ops::Range;

use super::{Byte, Dfa, EdgeSet, union};

fn bytes(range: Range<u16>) -> Byte {
    Byte { start: range.start, end: range.end }
}

fn assert_union(
    xs: &[(Byte, u8)],
    ys: &[(Byte, u8)],
    expected: &[(Byte, (Option<u8>, Option<u8>))],
) {
    let actual: Vec<_> = union(xs.iter().copied(), ys.iter().copied()).collect();
    assert_eq!(actual, expected);

    let actual: Vec<_> = union(ys.iter().copied(), xs.iter().copied()).collect();
    let expected: Vec<_> = expected.iter().map(|&(range, (x, y))| (range, (y, x))).collect();
    assert_eq!(actual, expected);
}

#[test]
fn union_empty_inputs() {
    assert_union(&[], &[], &[]);
    assert_union(
        &[(bytes(1..3), 1), (bytes(5..7), 2)],
        &[],
        &[(bytes(1..3), (Some(1), None)), (bytes(5..7), (Some(2), None))],
    );
}

#[test]
fn union_disjoint_and_adjacent_ranges() {
    assert_union(
        &[(bytes(1..3), 1), (bytes(6..8), 2)],
        &[(bytes(3..5), 3), (bytes(8..9), 4)],
        &[
            (bytes(1..3), (Some(1), None)),
            (bytes(3..5), (None, Some(3))),
            (bytes(6..8), (Some(2), None)),
            (bytes(8..9), (None, Some(4))),
        ],
    );
}

#[test]
fn union_equal_ranges() {
    assert_union(&[(bytes(1..4), 1)], &[(bytes(1..4), 2)], &[(bytes(1..4), (Some(1), Some(2)))]);
}

#[test]
fn union_partially_overlapping_ranges() {
    assert_union(
        &[(bytes(1..4), 1)],
        &[(bytes(3..6), 2)],
        &[
            (bytes(1..3), (Some(1), None)),
            (bytes(3..4), (Some(1), Some(2))),
            (bytes(4..6), (None, Some(2))),
        ],
    );
}

#[test]
fn union_contained_ranges() {
    assert_union(
        &[(bytes(0..8), 1)],
        &[(bytes(2..4), 2), (bytes(4..6), 3)],
        &[
            (bytes(0..2), (Some(1), None)),
            (bytes(2..4), (Some(1), Some(2))),
            (bytes(4..6), (Some(1), Some(3))),
            (bytes(6..8), (Some(1), None)),
        ],
    );
}

#[test]
fn union_matching_starts_and_ends() {
    assert_union(
        &[(bytes(0..4), 1), (bytes(6..10), 2)],
        &[(bytes(0..2), 3), (bytes(8..10), 4)],
        &[
            (bytes(0..2), (Some(1), Some(3))),
            (bytes(2..4), (Some(1), None)),
            (bytes(6..8), (Some(2), None)),
            (bytes(8..10), (Some(2), Some(4))),
        ],
    );
}

#[test]
fn union_uninit_boundary() {
    let uninit = Byte::UNINIT;
    assert_union(
        &[(Byte::uninit(), 1)],
        &[(bytes(255..uninit), 2), (bytes(uninit..uninit + 1), 3)],
        &[
            (bytes(0..255), (Some(1), None)),
            (bytes(255..uninit), (Some(1), Some(2))),
            (bytes(uninit..uninit + 1), (Some(1), Some(3))),
        ],
    );
}

#[test]
fn edge_set_union_coalesces_only_adjacent_ranges_with_equal_destinations() {
    let xs = EdgeSet::from_edges(vec![(bytes(0..4), 1), (bytes(8..10), 1)]);
    let ys = EdgeSet::from_edges(vec![(bytes(2..6), 2), (bytes(10..12), 3)]);
    let merged = xs.union(&ys, |_, y| if y == Some(3) { 8 } else { 7 });

    assert_eq!(
        merged.iter().collect::<Vec<_>>(),
        [(bytes(0..6), 7), (bytes(8..10), 7), (bytes(10..12), 8)],
    );
}

#[test]
fn dot_distinguishes_start_and_accept() {
    for dfa in [Dfa::<!, !>::from_edges(0, 1, &[(0, 0u8, 1)]), Dfa::unit()] {
        let dot = format!("{dfa:?}");
        assert!(dot.lines().any(|line| line == "    start [shape = point, style = invis]"));
        let start_edge = format!("    start -> {:?}", dfa.start);
        assert!(dot.lines().any(|line| line == start_edge));

        let accept_marker = format!("    {:?} [shape = doublecircle]", dfa.accept);
        let accept_markers: Vec<_> =
            dot.lines().filter(|line| line.contains("doublecircle")).collect();
        assert_eq!(accept_markers, [accept_marker.as_str()]);
    }
}
