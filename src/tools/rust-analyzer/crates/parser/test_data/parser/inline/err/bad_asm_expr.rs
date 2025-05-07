fn foo() {
    builtin#asm(
        label crashy = { return; }
    );
}
