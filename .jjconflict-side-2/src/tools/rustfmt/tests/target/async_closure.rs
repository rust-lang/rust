// rustfmt-edition: 2018

fn main() {
    let async_closure = async {
        let x = 3;
        x
    };

    let f = async /* comment */ {
        let x = 3;
        x
    };

    let g = async /* comment */ move {
        let x = 3;
        x
    };

    let f = |x| async {
        println!("hello, world");
    };
}
