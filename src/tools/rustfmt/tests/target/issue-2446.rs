enum Issue2446 {
    V {
        f: u8, // x
    },
}

enum Issue2446TrailingCommentsOnly {
    V { f: u8 /* */ },
}
