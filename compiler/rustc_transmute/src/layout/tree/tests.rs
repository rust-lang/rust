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
    }

    mod should_reject {
        use super::*;

        #[test]
        fn invisible_def() {
            let layout: Tree<Def, !, !> = Tree::def(Def::HasSafetyInvariants);
            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }

        #[test]
        fn invisible_def_in_seq_len_2() {
            let layout: Tree<Def, !, !> =
                Tree::def(Def::NoSafetyInvariants).then(Tree::def(Def::HasSafetyInvariants));
            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }

        #[test]
        fn invisible_def_in_seq_len_3() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants)
                .then(Tree::byte(0x00))
                .then(Tree::def(Def::HasSafetyInvariants));
            assert_eq!(
                layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)),
                Tree::uninhabited()
            );
        }
    }

    mod should_accept {
        use super::*;

        #[test]
        fn visible_def() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants);
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::unit());
        }

        #[test]
        fn visible_def_in_seq_len_2() {
            let layout: Tree<Def, !, !> =
                Tree::def(Def::NoSafetyInvariants).then(Tree::def(Def::NoSafetyInvariants));
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::unit());
        }

        #[test]
        fn visible_def_in_seq_len_3() {
            let layout: Tree<Def, !, !> = Tree::def(Def::NoSafetyInvariants)
                .then(Tree::byte(0x00))
                .then(Tree::def(Def::NoSafetyInvariants));
            assert_eq!(layout.prune(&|d| matches!(d, Def::HasSafetyInvariants)), Tree::byte(0x00));
        }
    }
}
