extern int liba_process(int v);
extern int liba_answer();
extern int libb_multiply(int a, int b);
extern int libb_greet();

int main() {
    if (liba_answer() != 42) return 1;
    if (liba_process(10) != 31) return 1;

    if (libb_multiply(6, 7) != 42) return 1;
    if (libb_greet() != 99) return 1;

    return 0;
}
