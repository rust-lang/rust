fn foo() {
    k#asm(
        label crashy = { return; }
    );
}
