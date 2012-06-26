// error-pattern: mismatched types

fn main() {
    let v = {mut g: [0]/~};

    fn f(&&v: {mut g: [const int]/~}) {
        v.g = [mut 3]/~
    }

    f(v);
}
