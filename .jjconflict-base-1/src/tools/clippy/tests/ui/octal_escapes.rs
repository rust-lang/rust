//@no-rustfix: overlapping suggestions
#![warn(clippy::octal_escapes)]

fn main() {
    let _bad1 = "\033[0m";
    //~^ octal_escapes
    let _bad2 = b"\033[0m";
    //~^ octal_escapes
    let _bad3 = "\\\033[0m";
    //~^ octal_escapes
    // maximum 3 digits (\012 is the escape)
    let _bad4 = "\01234567";
    //~^ octal_escapes
    let _bad5 = "\0\03";
    //~^ octal_escapes
    let _bad6 = "Text-\055\077-MoreText";
    //~^ octal_escapes
    //~| octal_escapes

    let _bad7 = "EvenMoreText-\01\02-ShortEscapes";
    //~^ octal_escapes
    //~| octal_escapes

    let _bad8 = "锈\01锈";
    //~^ octal_escapes
    let _bad9 = "锈\011锈";
    //~^ octal_escapes

    let _good1 = "\\033[0m";
    let _good2 = "\0\\0";
    let _good3 = "\0\0";
    let _good4 = "X\0\0X";
    let _good5 = "锈\0锈";
    let _good6 = "\0\\01";
}
