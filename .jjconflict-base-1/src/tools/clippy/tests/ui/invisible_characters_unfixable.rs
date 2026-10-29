//@no-rustfix
#![warn(clippy::invisible_characters)]
#![allow(dead_code)]

fn invisible() {
    print!(r"a ZWS >​< here");
    //~^ invisible_characters
    print!(r"a SHY >­< here");
    //~^ invisible_characters
    print!(r"a WJ >⁠< here");
    //~^ invisible_characters
    print!(r#"a ZWS >​< between hashes"#);
    //~^ invisible_characters
}

fn main() {}
