use super::*;

impl<T: Eq + Hash + Copy> TransitiveRelation<T> {
    /// A "best" parent in some sense. See `parents` and
    /// `postdom_upper_bound` for more details.
    fn postdom_parent(&self, a: T) -> Option<T> {
        self.mutual_immediate_postdominator(self.parents(a))
    }
}

#[test]
fn test_one_step() {
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "b");
    relation.add("a", "c");
    let relation = relation.freeze();
    assert!(relation.contains("a", "c"));
    assert!(relation.contains("a", "b"));
    assert!(!relation.contains("b", "a"));
    assert!(!relation.contains("a", "d"));
}

#[test]
fn test_many_steps() {
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "b");
    relation.add("a", "c");
    relation.add("a", "f");

    relation.add("b", "c");
    relation.add("b", "d");
    relation.add("b", "e");

    relation.add("e", "g");
    let relation = relation.freeze();

    assert!(relation.contains("a", "b"));
    assert!(relation.contains("a", "c"));
    assert!(relation.contains("a", "d"));
    assert!(relation.contains("a", "e"));
    assert!(relation.contains("a", "f"));
    assert!(relation.contains("a", "g"));

    assert!(relation.contains("b", "g"));

    assert!(!relation.contains("a", "x"));
    assert!(!relation.contains("b", "f"));
}

#[test]
fn mubs_triangle() {
    // a -> tcx
    //      ^
    //      |
    //      b
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "tcx");
    relation.add("b", "tcx");
    let relation = relation.freeze();
    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["tcx"]);
    assert_eq!(relation.parents("a"), vec!["tcx"]);
    assert_eq!(relation.parents("b"), vec!["tcx"]);
}

#[test]
fn mubs_best_choice1() {
    // 0 -> 1 <- 3
    // |    ^    |
    // |    |    |
    // +--> 2 <--+
    //
    // mubs(0,3) = [1]

    // This tests a particular state in the algorithm, in which we
    // need the second pare down call to get the right result (after
    // intersection, we have [1, 2], but 2 -> 1).

    let mut relation = TransitiveRelationBuilder::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("2", "1");

    relation.add("3", "1");
    relation.add("3", "2");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("0", "3"), vec!["2"]);
    assert_eq!(relation.parents("0"), vec!["2"]);
    assert_eq!(relation.parents("2"), vec!["1"]);
    assert!(relation.parents("1").is_empty());
}

#[test]
fn mubs_best_choice2() {
    // 0 -> 1 <- 3
    // |    |    |
    // |    v    |
    // +--> 2 <--+
    //
    // mubs(0,3) = [2]

    // Like the preceding test, but in this case intersection is [2,
    // 1], and hence we rely on the first pare down call.

    let mut relation = TransitiveRelationBuilder::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("1", "2");

    relation.add("3", "1");
    relation.add("3", "2");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("0", "3"), vec!["1"]);
    assert_eq!(relation.parents("0"), vec!["1"]);
    assert_eq!(relation.parents("1"), vec!["2"]);
    assert!(relation.parents("2").is_empty());
}

#[test]
fn mubs_no_best_choice() {
    // in this case, the intersection yields [1, 2], and the "pare
    // down" calls find nothing to remove.
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("3", "1");
    relation.add("3", "2");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("0", "3"), vec!["1", "2"]);
    assert_eq!(relation.parents("0"), vec!["1", "2"]);
    assert_eq!(relation.parents("3"), vec!["1", "2"]);
}

#[test]
fn mubs_best_choice_scc() {
    // in this case, 1 and 2 form a cycle; we pick arbitrarily (but
    // consistently).

    let mut relation = TransitiveRelationBuilder::default();
    relation.add("0", "1");
    relation.add("0", "2");

    relation.add("1", "2");
    relation.add("2", "1");

    relation.add("3", "1");
    relation.add("3", "2");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("0", "3"), vec!["1"]);
    assert_eq!(relation.parents("0"), vec!["1"]);
}

#[test]
fn pdub_crisscross() {
    // diagonal edges run left-to-right
    // a -> a1 -> x
    //   \/       ^
    //   /\       |
    // b -> b1 ---+

    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "a1");
    relation.add("a", "b1");
    relation.add("b", "a1");
    relation.add("b", "b1");
    relation.add("a1", "x");
    relation.add("b1", "x");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["a1", "b1"]);
    assert_eq!(relation.postdom_upper_bound("a", "b"), Some("x"));
    assert_eq!(relation.postdom_parent("a"), Some("x"));
    assert_eq!(relation.postdom_parent("b"), Some("x"));
}

#[test]
fn pdub_crisscross_more() {
    // diagonal edges run left-to-right
    // a -> a1 -> a2 -> a3 -> x
    //   \/    \/             ^
    //   /\    /\             |
    // b -> b1 -> b2 ---------+

    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "a1");
    relation.add("a", "b1");
    relation.add("b", "a1");
    relation.add("b", "b1");

    relation.add("a1", "a2");
    relation.add("a1", "b2");
    relation.add("b1", "a2");
    relation.add("b1", "b2");

    relation.add("a2", "a3");

    relation.add("a3", "x");
    relation.add("b2", "x");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["a1", "b1"]);
    assert_eq!(relation.minimal_upper_bounds("a1", "b1"), vec!["a2", "b2"]);
    assert_eq!(relation.postdom_upper_bound("a", "b"), Some("x"));

    assert_eq!(relation.postdom_parent("a"), Some("x"));
    assert_eq!(relation.postdom_parent("b"), Some("x"));
}

#[test]
fn pdub_lub() {
    // a -> a1 -> x
    //            ^
    //            |
    // b -> b1 ---+

    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "a1");
    relation.add("b", "b1");
    relation.add("a1", "x");
    relation.add("b1", "x");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["x"]);
    assert_eq!(relation.postdom_upper_bound("a", "b"), Some("x"));

    assert_eq!(relation.postdom_parent("a"), Some("a1"));
    assert_eq!(relation.postdom_parent("b"), Some("b1"));
    assert_eq!(relation.postdom_parent("a1"), Some("x"));
    assert_eq!(relation.postdom_parent("b1"), Some("x"));
}

#[test]
fn mubs_intermediate_node_on_one_side_only() {
    // a -> c -> d
    //           ^
    //           |
    //           b

    // "digraph { a -> c -> d; b -> d; }",
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("b", "d");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["d"]);
}

#[test]
fn mubs_scc_1() {
    // +-------------+
    // |    +----+   |
    // |    v    |   |
    // a -> c -> d <-+
    //           ^
    //           |
    //           b

    // "digraph { a -> c -> d; d -> c; a -> d; b -> d; }",
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "c");
    relation.add("a", "d");
    relation.add("b", "d");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["c"]);
}

#[test]
fn mubs_scc_2() {
    //      +----+
    //      v    |
    // a -> c -> d
    //      ^    ^
    //      |    |
    //      +--- b

    // "digraph { a -> c -> d; d -> c; b -> d; b -> c; }",
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "c");
    relation.add("b", "d");
    relation.add("b", "c");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["c"]);
}

#[test]
fn mubs_scc_3() {
    //      +---------+
    //      v         |
    // a -> c -> d -> e
    //           ^    ^
    //           |    |
    //           b ---+

    // "digraph { a -> c -> d -> e -> c; b -> d; b -> e; }",
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "e");
    relation.add("e", "c");
    relation.add("b", "d");
    relation.add("b", "e");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["c"]);
}

#[test]
fn mubs_scc_4() {
    //      +---------+
    //      v         |
    // a -> c -> d -> e
    // |         ^    ^
    // +---------+    |
    //                |
    //           b ---+

    // "digraph { a -> c -> d -> e -> c; a -> d; b -> e; }"
    let mut relation = TransitiveRelationBuilder::default();
    relation.add("a", "c");
    relation.add("c", "d");
    relation.add("d", "e");
    relation.add("e", "c");
    relation.add("a", "d");
    relation.add("b", "e");
    let relation = relation.freeze();

    assert_eq!(relation.minimal_upper_bounds("a", "b"), vec!["c"]);
}

#[test]
fn parent() {
    // An example that was misbehaving in the compiler.
    //
    // 4 -> 1 -> 3
    //   \  |   /
    //    \ v  /
    // 2 -> 0
    //
    // plus a bunch of self-loops
    //
    // Here `->` represents `<=` and `0` is `'static`.

    let pairs = vec![
        (2, /*->*/ 0),
        (2, /*->*/ 2),
        (0, /*->*/ 0),
        (0, /*->*/ 0),
        (1, /*->*/ 0),
        (1, /*->*/ 1),
        (3, /*->*/ 0),
        (3, /*->*/ 3),
        (4, /*->*/ 0),
        (4, /*->*/ 1),
        (1, /*->*/ 3),
    ];

    let mut relation = TransitiveRelationBuilder::default();
    for (a, b) in pairs {
        relation.add(a, b);
    }
    let relation = relation.freeze();

    let p = relation.postdom_parent(3);
    assert_eq!(p, Some(0));
}
