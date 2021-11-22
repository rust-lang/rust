#![warn(clippy::octal_escapes)]

fn main() {
    let _bad1 = "\033[0m";
    let _bad2 = b"\033[0m";
    let _bad3 = "\\\033[0m";
    // maximum 3 digits (\012 is the escape)
    let _bad4 = "\01234567";
    let _bad5 = "\0\03";
    let _bad6 = "Text-\055\077-MoreText";
    let _bad7 = "EvenMoreText-\01\02-ShortEscapes";
    let _bad8 = "锈\01锈";
    let _bad9 = "锈\011锈";

    let _good1 = "\\033[0m";
    let _good2 = "\0\\0";
    let _good3 = "\0\0";
    let _good4 = "X\0\0X";
    let _good5 = "锈\0锈";
}
