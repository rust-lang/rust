//@ edition:2021
//@ build-pass

fn main() {
    let _ = async {
        let mut s = (String::new(),);
        s.0.push_str("abc");
        std::mem::drop(s);
        async {}.await;
    };
}
