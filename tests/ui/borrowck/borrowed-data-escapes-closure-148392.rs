// Test for issue #148392
// Provides helpful explanations even for anonymous references
// under the scenario of escaping closure

#![allow(unused)]

fn main() {
    let a = 0;
    let mut b = None;
    move || {
        b = Some(&a); //~ ERROR borrowed data escapes outside of closure
    };
}
