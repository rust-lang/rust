int
do_twice(int (*fn)(int), int arg)
{
    return fn(arg) + fn(arg);
}
