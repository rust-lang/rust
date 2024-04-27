//@no-rustfix: overlapping suggestions
#![warn(clippy::octal_escapes)]

fn main() {
    let _bad1 = "\033[0m";
    //~^ ERROR: octal-looking escape in string literal
    let _bad2 = b"\033[0m";
    //~^ ERROR: octal-looking escape in byte string literal
    let _bad3 = "\\\033[0m";
    //~^ ERROR: octal-looking escape in string literal
    // maximum 3 digits (\012 is the escape)
    let _bad4 = "\01234567";
    //~^ ERROR: octal-looking escape in string literal
    let _bad5 = "\0\03";
    //~^ ERROR: octal-looking escape in string literal
    let _bad6 = "Text-\055\077-MoreText";
    //~^ ERROR: octal-looking escape in string literal
    let _bad7 = "EvenMoreText-\01\02-ShortEscapes";
    //~^ ERROR: octal-looking escape in string literal
    let _bad8 = "锈\01锈";
    //~^ ERROR: octal-looking escape in string literal
    let _bad9 = "锈\011锈";
    //~^ ERROR: octal-looking escape in string literal

    let _good1 = "\\033[0m";
    let _good2 = "\0\\0";
    let _good3 = "\0\0";
    let _good4 = "X\0\0X";
    let _good5 = "锈\0锈";
    let _good6 = "\0\\01";
}
