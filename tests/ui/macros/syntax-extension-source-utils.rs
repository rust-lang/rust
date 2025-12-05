//@ run-pass
#![allow(stable_features)]


pub mod m1 {
    pub mod m2 {
        pub fn where_am_i() -> String {
            (module_path!()).to_string()
        }
    }
}

macro_rules! indirect_line { () => ( line!() ) }

pub fn main() {
    assert_eq!(line!(), 16);
    assert_eq!(column!(), 16);
    assert_eq!(indirect_line!(), 18);
    assert!(file!().ends_with("syntax-extension-source-utils.rs"));
    assert_eq!(stringify!((2*3) + 5).to_string(), "(2*3) + 5".to_string());
    assert!(include!("syntax-extension-source-utils-files/includeme.\
                      fragment").to_string()
           == "victory robot 6".to_string());

    assert!(
        include_str!("syntax-extension-source-utils-files/includeme.\
                      fragment").to_string()
        .starts_with("/* this is for "));
    assert!(
        include_bytes!("syntax-extension-source-utils-files/includeme.fragment")
        [1] == (42 as u8)); // '*'
    // The Windows tests are wrapped in an extra module for some reason
    assert!(m1::m2::where_am_i().ends_with("m1::m2"));

    assert_eq!((35, "(2*3) + 5"), (line!(), stringify!((2*3) + 5)));
}
