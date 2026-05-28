//@ edition:2021
//@ check-pass
#![warn(unused)]

fn main() {
    let t = (String::from("Hello"), String::from("World"));
    let g = (String::from("Mr"), String::from("Goose"));

    let a = || {
        let (_, g2) = g;
        //~^ WARN unused variable: `g2`
        let c = ||  {
            let (_, t2) = t;
            //~^ WARN unused variable: `t2`
        };

        c();
    };

    a();
}
