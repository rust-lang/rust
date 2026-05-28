#![allow(warnings)]
#![feature(rustc_attrs)]

fn main() {
}

fn nll_fail() {
    let mut data = vec!['a', 'b', 'c'];
    let slice = &mut data;
    capitalize(slice);
    data.push('d');
    //~^ ERROR [E0499]
    data.push('e');
    //~^ ERROR [E0499]
    data.push('f');
    //~^ ERROR [E0499]
    capitalize(slice);
}

fn nll_ok() {
    let mut data = vec!['a', 'b', 'c'];
    let slice = &mut data;
    capitalize(slice);
    data.push('d');
    data.push('e');
    data.push('f');
}

fn capitalize(_: &mut [char]) {
}
