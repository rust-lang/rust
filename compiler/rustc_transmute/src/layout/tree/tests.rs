use super::Tree;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum Def {
    Visible,
    Invisible,
}

impl super::Def for Def {}

mod prune {
    use super::*;

    mod should_simplify {
        use super::*;

        #[test]
        fn seq_1() {
            let layout: Tree<Def, !> = Tree::def(Def::Visible).then(Tree::from_bits(0x00));
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::from_bits(0x00));
        }

        #[test]
        fn seq_2() {
            let layout: Tree<Def, !> =
                Tree::from_bits(0x00).then(Tree::def(Def::Visible)).then(Tree::from_bits(0x01));

            assert_eq!(
                layout.prune(&|d| matches!(d, Def::Visible)),
                Tree::from_bits(0x00).then(Tree::from_bits(0x01))
            );
        }
    }

    mod should_reject {
        use super::*;

        #[test]
        fn invisible_def() {
            let layout: Tree<Def, !> = Tree::def(Def::Invisible);
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::uninhabited());
        }

        #[test]
        fn invisible_def_in_seq_len_2() {
            let layout: Tree<Def, !> = Tree::def(Def::Visible).then(Tree::def(Def::Invisible));
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::uninhabited());
        }

        #[test]
        fn invisible_def_in_seq_len_3() {
            let layout: Tree<Def, !> =
                Tree::def(Def::Visible).then(Tree::from_bits(0x00)).then(Tree::def(Def::Invisible));
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::uninhabited());
        }
    }

    mod should_accept {
        use super::*;

        #[test]
        fn visible_def() {
            let layout: Tree<Def, !> = Tree::def(Def::Visible);
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::unit());
        }

        #[test]
        fn visible_def_in_seq_len_2() {
            let layout: Tree<Def, !> = Tree::def(Def::Visible).then(Tree::def(Def::Visible));
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::unit());
        }

        #[test]
        fn visible_def_in_seq_len_3() {
            let layout: Tree<Def, !> =
                Tree::def(Def::Visible).then(Tree::from_bits(0x00)).then(Tree::def(Def::Visible));
            assert_eq!(layout.prune(&|d| matches!(d, Def::Visible)), Tree::from_bits(0x00));
        }
    }
}
