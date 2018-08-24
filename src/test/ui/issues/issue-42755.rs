#![cfg_attr(stage0, feature(macro_vis_matcher))]

macro_rules! foo {
    ($($p:vis)*) => {} //~ ERROR repetition matches empty token tree
}

foo!(a);
