//@ known-bug: #140281

macro_rules! foo {
    ($x:expr) => { $x }
}

fn main() {
    let t = vec![
        /// ‮test⁦ RTL in doc in vec!
        //  ICE (Sadly)
        1
    ];

        foo!(
        /// ‮test⁦ RTL in doc in macro
        1
    );
}
