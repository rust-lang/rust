fn issue982() {
    const SOME_CONSTANT: u32 =
        // Explanation why SOME_CONSTANT needs FLAG_A to be set.
        FLAG_A |
        // Explanation why SOME_CONSTANT needs FLAG_B to be set.
        FLAG_B |
        // Explanation why SOME_CONSTANT needs FLAG_C to be set.
        FLAG_C;
}
