extern "C" int cc_plus_one_cxx(int *arg);
extern "C" int cc_plus_one_cxx_asm(int *arg);

int cc_plus_one_cxx(int *arg) {
    return *arg + 1;
}

int cc_plus_one_cxx_asm(int *arg) {
    int value = 0;

    asm volatile ( "    movl (%1), %0\n"
                   "    inc %0\n"
                   "    jmp 1f\n"
                   "    retq\n"  // never executed, but a shortcut to determine how
                                 // the assembler deals with `ret` instructions
                   "1:\n"
                   : "=r"(value)
                   : "r"(arg) );

    return value;
}
