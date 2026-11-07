use std::ops::Range;

use super::{Byte, Dfa, EdgeSet, Reference, State, union};

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
fn edge_set_from_edges_accepts_empty_input() {
    assert_eq!(EdgeSet::<u8>::from_edges(vec![]), EdgeSet::empty());
}

#[test]
fn edge_set_from_edges_sorts_valid_ranges() {
    let uninit = Byte::UNINIT;
    let edges = EdgeSet::from_edges(vec![
        (bytes(uninit..uninit + 1), 3),
        (bytes(4..6), 2),
        (bytes(0..4), 1),
    ]);
    assert_eq!(
        edges.iter().collect::<Vec<_>>(),
        [(bytes(0..4), 1), (bytes(4..6), 2), (bytes(uninit..uninit + 1), 3)],
    );
}

#[test]
#[should_panic(expected = "invalid byte edge range")]
fn edge_set_from_edges_rejects_empty_range() {
    EdgeSet::from_edges(vec![(bytes(2..2), 0)]);
}

#[test]
#[should_panic(expected = "invalid byte edge range")]
fn edge_set_from_edges_rejects_reversed_range() {
    EdgeSet::from_edges(vec![(bytes(4..2), 0)]);
}

#[test]
#[should_panic(expected = "invalid byte edge range")]
fn edge_set_from_edges_rejects_range_past_uninit() {
    EdgeSet::from_edges(vec![(bytes(0..Byte::UNINIT + 2), 0)]);
}

#[test]
#[should_panic(expected = "byte edge ranges overlap")]
fn edge_set_from_edges_rejects_overlapping_ranges() {
    EdgeSet::from_edges(vec![(bytes(3..5), 0), (bytes(1..4), 1)]);
}

fn reference(region: usize) -> Reference<usize, ()> {
    Reference { region, is_mut: false, referent: (), referent_size: 0, referent_align: 1 }
}

#[test]
fn concat_preserves_reference_edges() {
    let first_ref = reference(1);
    let second_ref = reference(2);
    let first = Dfa::from_ref(first_ref);
    let second = Dfa::from_ref(second_ref).concat(Dfa::from_byte(7u8.into()));
    let start = first.start;
    let boundary = first.accept;
    let second_start = second.start;
    let after_second_ref = second.refs_from(second.start).next().unwrap().1;
    let accept = second.accept;

    let concatenated = first.concat(second);

    assert_eq!(concatenated.start, start);
    assert_eq!(concatenated.accept, accept);
    assert_eq!(concatenated.refs_from(start).collect::<Vec<_>>(), [(first_ref, boundary)]);
    assert_eq!(
        concatenated.refs_from(boundary).collect::<Vec<_>>(),
        [(second_ref, after_second_ref)],
    );
    assert_eq!(
        concatenated.bytes_from(after_second_ref).collect::<Vec<_>>(),
        [(7u8.into(), accept)]
    );
    assert!(!concatenated.transitions.contains_key(&second_start));
}

#[test]
fn unit_is_concat_identity_for_reference_graph() {
    let dfa = Dfa::from_ref(reference(1)).concat(Dfa::from_byte(7u8.into()));
    assert_eq!(Dfa::unit().concat(dfa.clone()), dfa);
    assert_eq!(dfa.clone().concat(Dfa::unit()), dfa);
}

#[test]
fn union_merges_shared_and_distinct_reference_edges_in_order() {
    let shared_ref = reference(1);
    let a_only_ref = reference(2);
    let b_only_ref = reference(3);
    let mut a = Dfa::from_ref(shared_ref).concat(Dfa::from_byte(1u8.into()));
    let mut b = Dfa::from_ref(shared_ref).concat(Dfa::from_byte(5u8.into()));
    let a_continuation = a.refs_from(a.start).next().unwrap().1;
    let b_continuation = b.refs_from(b.start).next().unwrap().1;
    a.transitions.get_mut(&a.start).unwrap().ref_transitions.insert(a_only_ref, a_continuation);
    b.transitions.get_mut(&b.start).unwrap().ref_transitions.insert(b_only_ref, b_continuation);

    let merged = a.union(b, State::new);
    let refs: Vec<_> = merged.refs_from(merged.start).collect();

    assert_eq!(
        refs.iter().map(|&(r, _)| r).collect::<Vec<_>>(),
        [shared_ref, a_only_ref, b_only_ref]
    );
    assert_eq!(
        merged.bytes_from(refs[0].1).collect::<Vec<_>>(),
        [(1u8.into(), merged.accept), (5u8.into(), merged.accept)],
    );
    assert_eq!(merged.bytes_from(refs[1].1).collect::<Vec<_>>(), [(1u8.into(), merged.accept)]);
    assert_eq!(merged.bytes_from(refs[2].1).collect::<Vec<_>>(), [(5u8.into(), merged.accept)]);
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
