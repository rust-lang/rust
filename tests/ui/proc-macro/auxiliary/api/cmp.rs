use proc_macro::{Punct, Spacing};

pub fn test() {
    test_punct_eq();
}

fn test_punct_eq() {
    let colon_alone = Punct::new(':', Spacing::Alone);
    assert_eq!(colon_alone, ':');
    let colon_joint = Punct::new(':', Spacing::Joint);
    assert_eq!(colon_joint, ':');
}
