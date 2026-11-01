use super::Tree;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
enum Def {
    NoSafetyInvariants,
    HasSafetyInvariants,
}

impl super::Def for Def {
    fn has_safety_invariants(&self) -> bool {
        self == &Self::HasSafetyInvariants
    }
}

mod prune {
    use super::*;

    mod should_simplify {
        use super::*;

        #[test]
        fn seq_1() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants).then(Tree::byte(0x00));
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::byte(0x00));
        }

        #[test]
        fn seq_2() {
            let layout: Tree<Def, !, !> =
                Tree::byte(0x00).then(Tree::def(Def::NoSafetyInvariants)).then(Tree::byte(0x01));

            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::byte(0x00).then(Tree::byte(0x01))
            );
        }

        #[test]
        fn single_retained_alternative() {
            let layout: Tree<Def, !, !> = Tree::alt([
                Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x00)),
                Tree::def(Def::NoSafetyInvariants).then(Tree::byte(0x01)),
                Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x02)),
            ]);

            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::byte(0x01));
        }

        #[test]
        fn mixed_nested_alternatives_in_seq() {
            let layout: Tree<Def, !, !> = Tree::seq([
                Tree::byte(0x00),
                Tree::Alt(vec![
                    Tree::def(Def::NoSafetyInvariants).then(Tree::byte(0x01)),
                    Tree::Alt(vec![
                        Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x02)),
                        Tree::def(Def::NoSafetyInvariants).then(Tree::byte(0x03)),
                    ]),
                ]),
                Tree::byte(0x04),
            ]);

            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::seq([
                    Tree::byte(0x00),
                    Tree::alt([Tree::byte(0x01), Tree::byte(0x03)]),
                    Tree::byte(0x04),
                ])
            );
        }
    }

    mod should_reject {
        use super::*;

        #[test]
        fn def_with_safety_invariants() {
            let layout: Tree<Def, !, !> = Tree::def(Def::HasSafetyInvariants);
            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }

        #[test]
        fn def_with_safety_invariants_in_seq_len_2() {
            let layout: Tree<Def, !, !> =
                Tree::def(Def::NoSafetyInvariants).then(Tree::def(Def::HasSafetyInvariants));
            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }

        #[test]
        fn def_with_safety_invariants_in_seq_len_3() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants)
                .then(Tree::byte(0x00))
                .then(Tree::def(Def::HasSafetyInvariants));
            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }

        #[test]
        fn all_alternatives_pruned() {
            let layout: Tree<Def, !, !> = Tree::alt([
                Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x00)),
                Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x01)),
            ]);

            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }

        #[test]
        fn all_alternatives_pruned_in_seq() {
            let layout: Tree<Def, !, !> = Tree::seq([
                Tree::byte(0x00),
                Tree::alt([
                    Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x01)),
                    Tree::def(Def::HasSafetyInvariants).then(Tree::byte(0x02)),
                ]),
                Tree::byte(0x03),
            ]);

            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }
    }

    mod should_accept {
        use super::*;

        #[test]
        fn def_without_safety_invariants() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants);
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::unit());
        }

        #[test]
        fn def_without_safety_invariants_in_seq_len_2() {
            let layout: Tree<Def, !, !> =
                Tree::def(Def::NoSafetyInvariants).then(Tree::def(Def::NoSafetyInvariants));
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::unit());
        }

        #[test]
        fn def_without_safety_invariants_in_seq_len_3() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants)
                .then(Tree::byte(0x00))
                .then(Tree::def(Def::NoSafetyInvariants));
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::byte(0x00));
        }
    }
}
