// (#77273) These characters are in the general categories of
// "Uppercase/Lowercase Letter",
// but casing operations map them to themselves.
// Therefore, we do not warn about casing
// (but do warn about uncommon codepoints).

//@ check-pass

#![allow(unused)]

struct 𝕟𝕠𝕥𝕒𝕔𝕒𝕞𝕖𝕝;
//~^ WARN identifier contains 9 non normalized (NFKC) characters

struct 𝕟𝕠𝕥_𝕒_𝕔𝕒𝕞𝕖𝕝;
//~^ WARN identifier contains 9 non normalized (NFKC) characters

static 𝗻𝗼𝗻𝘂𝗽𝗽𝗲𝗿𝗰𝗮𝘀𝗲: i32 = 1;
//~^ WARN identifier contains 12 non normalized (NFKC) characters

fn main() {
    let 𝓢𝓝𝓐𝓐𝓐𝓐𝓚𝓔𝓢 = 1;
    //~^ WARN identifier contains 9 non normalized (NFKC) characters
}
