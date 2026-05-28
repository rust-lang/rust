macro_rules! many_args {
    ([$($t:tt)*]#$($h:tt)*) => {
        many_args!{[$($t)*$($t)*]$($h)*}
    };
    ([$($t:tt)*]) => {
        fn _f($($t: ()),*) {} //~ ERROR function can not have more than 65535 arguments
    }
}

many_args!{[_]########## ######}

fn main() {}
