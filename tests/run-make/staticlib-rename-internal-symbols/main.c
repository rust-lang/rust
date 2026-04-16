extern int my_add(int a, int b);
extern unsigned long my_hash_lookup(unsigned long key);
extern int call_internal(void);
extern int my_safe_div(int a, int b);

int main() {
    if (my_add(10, 20) != 30)
        return 1;
    if (my_hash_lookup(5) != 5UL * 2654435761UL)
        return 1;
    if (call_internal() != 42)
        return 1;
    if (my_safe_div(100, 5) != 20)
        return 1;
    if (my_safe_div(100, 0) != -1)
        return 1;
    return 0;
}
