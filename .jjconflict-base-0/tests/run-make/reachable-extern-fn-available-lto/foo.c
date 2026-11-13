extern void foo();
extern char FOO_STATIC;

int main() {
    foo();
    return (int)FOO_STATIC;
}
