// xfail-pretty

use std;
import std::str;

fn test(actual: str, expected: str) {
    log actual;
    log expected;
    assert (str::eq(actual, expected));
}

fn main() {
    test(#fmt("hello %d friends and %s things", 10, "formatted"),
         "hello 10 friends and formatted things");

    test(#fmt("test"), "test");

    // a quadratic optimization in LLVM (jump-threading) makes this test a
    // bit slow to compile unless we break it up
    part1();
    part2();
    part3();
    part4();
    part5();
    part6();
}

fn part1() {
    // Simple tests for types

    test(#fmt("%d", 1), "1");
    test(#fmt("%i", 2), "2");
    test(#fmt("%i", -1), "-1");
    test(#fmt("%u", 10u), "10");
    test(#fmt("%s", "test"), "test");
    test(#fmt("%b", true), "true");
    test(#fmt("%b", false), "false");
    test(#fmt("%c", 'A'), "A");
    test(#fmt("%x", 0xff_u), "ff");
    test(#fmt("%X", 0x12ab_u), "12AB");
    test(#fmt("%o", 10u), "12");
    test(#fmt("%t", 0b11010101_u), "11010101");
    // 32-bit limits

    test(#fmt("%i", -2147483648), "-2147483648");
    test(#fmt("%i", 2147483647), "2147483647");
    test(#fmt("%u", 4294967295u), "4294967295");
    test(#fmt("%x", 0xffffffff_u), "ffffffff");
    test(#fmt("%o", 0xffffffff_u), "37777777777");
    test(#fmt("%t", 0xffffffff_u), "11111111111111111111111111111111");
}
fn part2() {
    // Widths

    test(#fmt("%1d", 500), "500");
    test(#fmt("%10d", 500), "       500");
    test(#fmt("%10d", -500), "      -500");
    test(#fmt("%10u", 500u), "       500");
    test(#fmt("%10s", "test"), "      test");
    test(#fmt("%10b", true), "      true");
    test(#fmt("%10x", 0xff_u), "        ff");
    test(#fmt("%10X", 0xff_u), "        FF");
    test(#fmt("%10o", 10u), "        12");
    test(#fmt("%10t", 0xff_u), "  11111111");
    test(#fmt("%10c", 'A'), "         A");
    // Left justify

    test(#fmt("%-10d", 500), "500       ");
    test(#fmt("%-10d", -500), "-500      ");
    test(#fmt("%-10u", 500u), "500       ");
    test(#fmt("%-10s", "test"), "test      ");
    test(#fmt("%-10b", true), "true      ");
    test(#fmt("%-10x", 0xff_u), "ff        ");
    test(#fmt("%-10X", 0xff_u), "FF        ");
    test(#fmt("%-10o", 10u), "12        ");
    test(#fmt("%-10t", 0xff_u), "11111111  ");
    test(#fmt("%-10c", 'A'), "A         ");
}

fn part3() {
    // Precision

    test(#fmt("%.d", 0), "");
    test(#fmt("%.u", 0u), "");
    test(#fmt("%.x", 0u), "");
    test(#fmt("%.t", 0u), "");
    test(#fmt("%.d", 10), "10");
    test(#fmt("%.d", -10), "-10");
    test(#fmt("%.u", 10u), "10");
    test(#fmt("%.s", "test"), "");
    test(#fmt("%.x", 127u), "7f");
    test(#fmt("%.o", 10u), "12");
    test(#fmt("%.t", 3u), "11");
    test(#fmt("%.c", 'A'), "A");
    test(#fmt("%.0d", 0), "");
    test(#fmt("%.0u", 0u), "");
    test(#fmt("%.0x", 0u), "");
    test(#fmt("%.0t", 0u), "");
    test(#fmt("%.0d", 10), "10");
    test(#fmt("%.0d", -10), "-10");
    test(#fmt("%.0u", 10u), "10");
    test(#fmt("%.0s", "test"), "");
    test(#fmt("%.0x", 127u), "7f");
    test(#fmt("%.0o", 10u), "12");
    test(#fmt("%.0t", 3u), "11");
    test(#fmt("%.0c", 'A'), "A");
    test(#fmt("%.1d", 0), "0");
    test(#fmt("%.1u", 0u), "0");
    test(#fmt("%.1x", 0u), "0");
    test(#fmt("%.1t", 0u), "0");
    test(#fmt("%.1d", 10), "10");
    test(#fmt("%.1d", -10), "-10");
    test(#fmt("%.1u", 10u), "10");
    test(#fmt("%.1s", "test"), "t");
    test(#fmt("%.1x", 127u), "7f");
    test(#fmt("%.1o", 10u), "12");
    test(#fmt("%.1t", 3u), "11");
    test(#fmt("%.1c", 'A'), "A");
}
fn part4() {
    test(#fmt("%.5d", 0), "00000");
    test(#fmt("%.5u", 0u), "00000");
    test(#fmt("%.5x", 0u), "00000");
    test(#fmt("%.5t", 0u), "00000");
    test(#fmt("%.5d", 10), "00010");
    test(#fmt("%.5d", -10), "-00010");
    test(#fmt("%.5u", 10u), "00010");
    test(#fmt("%.5s", "test"), "test");
    test(#fmt("%.5x", 127u), "0007f");
    test(#fmt("%.5o", 10u), "00012");
    test(#fmt("%.5t", 3u), "00011");
    test(#fmt("%.5c", 'A'), "A");
    // Bool precision. I'm not sure if it's good or bad to have bool
    // conversions support precision - it's not standard printf so we
    // can do whatever. For now I'm making it behave the same as string
    // conversions.

    test(#fmt("%.b", true), "");
    test(#fmt("%.0b", true), "");
    test(#fmt("%.1b", true), "t");
}

fn part5() {
    // Explicit + sign. Only for signed conversions

    test(#fmt("%+d", 0), "+0");
    test(#fmt("%+d", 1), "+1");
    test(#fmt("%+d", -1), "-1");
    // Leave space for sign

    test(#fmt("% d", 0), " 0");
    test(#fmt("% d", 1), " 1");
    test(#fmt("% d", -1), "-1");
    // Plus overrides space

    test(#fmt("% +d", 0), "+0");
    test(#fmt("%+ d", 0), "+0");
    // 0-padding

    test(#fmt("%05d", 0), "00000");
    test(#fmt("%05d", 1), "00001");
    test(#fmt("%05d", -1), "-0001");
    test(#fmt("%05u", 1u), "00001");
    test(#fmt("%05x", 127u), "0007f");
    test(#fmt("%05X", 127u), "0007F");
    test(#fmt("%05o", 10u), "00012");
    test(#fmt("%05t", 3u), "00011");
    // 0-padding a string is undefined but glibc does this:

    test(#fmt("%05s", "test"), " test");
    test(#fmt("%05c", 'A'), "    A");
    test(#fmt("%05b", true), " true");
    // Left-justify overrides 0-padding

    test(#fmt("%-05d", 0), "0    ");
    test(#fmt("%-05d", 1), "1    ");
    test(#fmt("%-05d", -1), "-1   ");
    test(#fmt("%-05u", 1u), "1    ");
    test(#fmt("%-05x", 127u), "7f   ");
    test(#fmt("%-05X", 127u), "7F   ");
    test(#fmt("%-05o", 10u), "12   ");
    test(#fmt("%-05t", 3u), "11   ");
    test(#fmt("%-05s", "test"), "test ");
    test(#fmt("%-05c", 'A'), "A    ");
    test(#fmt("%-05b", true), "true ");
}
fn part6() {
    // Precision overrides 0-padding

    test(#fmt("%06.5d", 0), " 00000");
    test(#fmt("%06.5u", 0u), " 00000");
    test(#fmt("%06.5x", 0u), " 00000");
    test(#fmt("%06.5d", 10), " 00010");
    test(#fmt("%06.5d", -10), "-00010");
    test(#fmt("%06.5u", 10u), " 00010");
    test(#fmt("%06.5s", "test"), "  test");
    test(#fmt("%06.5c", 'A'), "     A");
    test(#fmt("%06.5x", 127u), " 0007f");
    test(#fmt("%06.5X", 127u), " 0007F");
    test(#fmt("%06.5o", 10u), " 00012");
    // Signed combinations

    test(#fmt("% 5d", 1), "    1");
    test(#fmt("% 5d", -1), "   -1");
    test(#fmt("%+5d", 1), "   +1");
    test(#fmt("%+5d", -1), "   -1");
    test(#fmt("% 05d", 1), " 0001");
    test(#fmt("% 05d", -1), "-0001");
    test(#fmt("%+05d", 1), "+0001");
    test(#fmt("%+05d", -1), "-0001");
    test(#fmt("%- 5d", 1), " 1   ");
    test(#fmt("%- 5d", -1), "-1   ");
    test(#fmt("%-+5d", 1), "+1   ");
    test(#fmt("%-+5d", -1), "-1   ");
    test(#fmt("%- 05d", 1), " 1   ");
    test(#fmt("%- 05d", -1), "-1   ");
    test(#fmt("%-+05d", 1), "+1   ");
    test(#fmt("%-+05d", -1), "-1   ");
}