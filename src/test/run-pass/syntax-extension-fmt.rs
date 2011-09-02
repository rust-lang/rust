use std;
import std::str;

fn test(actual: &istr, expected: &istr) {
    log actual;
    log expected;
    assert (str::eq(actual, expected));
}

fn main() {
    test(#ifmt[~"hello %d friends and %s things", 10, ~"formatted"],
         ~"hello 10 friends and formatted things");

    test(#ifmt[~"test"], ~"test");

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

    test(#ifmt[~"%d", 1], ~"1");
    test(#ifmt[~"%i", 2], ~"2");
    test(#ifmt[~"%i", -1], ~"-1");
    test(#ifmt[~"%u", 10u], ~"10");
    test(#ifmt[~"%s", ~"test"], ~"test");
    test(#ifmt[~"%b", true], ~"true");
    test(#ifmt[~"%b", false], ~"false");
    test(#ifmt[~"%c", 'A'], ~"A");
    test(#ifmt[~"%x", 0xff_u], ~"ff");
    test(#ifmt[~"%X", 0x12ab_u], ~"12AB");
    test(#ifmt[~"%o", 10u], ~"12");
    test(#ifmt[~"%t", 0b11010101_u], ~"11010101");
    // 32-bit limits

    test(#ifmt[~"%i", -2147483648], ~"-2147483648");
    test(#ifmt[~"%i", 2147483647], ~"2147483647");
    test(#ifmt[~"%u", 4294967295u], ~"4294967295");
    test(#ifmt[~"%x", 0xffffffff_u], ~"ffffffff");
    test(#ifmt[~"%o", 0xffffffff_u], ~"37777777777");
    test(#ifmt[~"%t", 0xffffffff_u], ~"11111111111111111111111111111111");
}
fn part2() {
    // Widths

    test(#ifmt[~"%1d", 500], ~"500");
    test(#ifmt[~"%10d", 500], ~"       500");
    test(#ifmt[~"%10d", -500], ~"      -500");
    test(#ifmt[~"%10u", 500u], ~"       500");
    test(#ifmt[~"%10s", ~"test"], ~"      test");
    test(#ifmt[~"%10b", true], ~"      true");
    test(#ifmt[~"%10x", 0xff_u], ~"        ff");
    test(#ifmt[~"%10X", 0xff_u], ~"        FF");
    test(#ifmt[~"%10o", 10u], ~"        12");
    test(#ifmt[~"%10t", 0xff_u], ~"  11111111");
    test(#ifmt[~"%10c", 'A'], ~"         A");
    // Left justify

    test(#ifmt[~"%-10d", 500], ~"500       ");
    test(#ifmt[~"%-10d", -500], ~"-500      ");
    test(#ifmt[~"%-10u", 500u], ~"500       ");
    test(#ifmt[~"%-10s", ~"test"], ~"test      ");
    test(#ifmt[~"%-10b", true], ~"true      ");
    test(#ifmt[~"%-10x", 0xff_u], ~"ff        ");
    test(#ifmt[~"%-10X", 0xff_u], ~"FF        ");
    test(#ifmt[~"%-10o", 10u], ~"12        ");
    test(#ifmt[~"%-10t", 0xff_u], ~"11111111  ");
    test(#ifmt[~"%-10c", 'A'], ~"A         ");
}

fn part3() {
    // Precision

    test(#ifmt[~"%.d", 0], ~"");
    test(#ifmt[~"%.u", 0u], ~"");
    test(#ifmt[~"%.x", 0u], ~"");
    test(#ifmt[~"%.t", 0u], ~"");
    test(#ifmt[~"%.d", 10], ~"10");
    test(#ifmt[~"%.d", -10], ~"-10");
    test(#ifmt[~"%.u", 10u], ~"10");
    test(#ifmt[~"%.s", ~"test"], ~"");
    test(#ifmt[~"%.x", 127u], ~"7f");
    test(#ifmt[~"%.o", 10u], ~"12");
    test(#ifmt[~"%.t", 3u], ~"11");
    test(#ifmt[~"%.c", 'A'], ~"A");
    test(#ifmt[~"%.0d", 0], ~"");
    test(#ifmt[~"%.0u", 0u], ~"");
    test(#ifmt[~"%.0x", 0u], ~"");
    test(#ifmt[~"%.0t", 0u], ~"");
    test(#ifmt[~"%.0d", 10], ~"10");
    test(#ifmt[~"%.0d", -10], ~"-10");
    test(#ifmt[~"%.0u", 10u], ~"10");
    test(#ifmt[~"%.0s", ~"test"], ~"");
    test(#ifmt[~"%.0x", 127u], ~"7f");
    test(#ifmt[~"%.0o", 10u], ~"12");
    test(#ifmt[~"%.0t", 3u], ~"11");
    test(#ifmt[~"%.0c", 'A'], ~"A");
    test(#ifmt[~"%.1d", 0], ~"0");
    test(#ifmt[~"%.1u", 0u], ~"0");
    test(#ifmt[~"%.1x", 0u], ~"0");
    test(#ifmt[~"%.1t", 0u], ~"0");
    test(#ifmt[~"%.1d", 10], ~"10");
    test(#ifmt[~"%.1d", -10], ~"-10");
    test(#ifmt[~"%.1u", 10u], ~"10");
    test(#ifmt[~"%.1s", ~"test"], ~"t");
    test(#ifmt[~"%.1x", 127u], ~"7f");
    test(#ifmt[~"%.1o", 10u], ~"12");
    test(#ifmt[~"%.1t", 3u], ~"11");
    test(#ifmt[~"%.1c", 'A'], ~"A");
}
fn part4() {
    test(#ifmt[~"%.5d", 0], ~"00000");
    test(#ifmt[~"%.5u", 0u], ~"00000");
    test(#ifmt[~"%.5x", 0u], ~"00000");
    test(#ifmt[~"%.5t", 0u], ~"00000");
    test(#ifmt[~"%.5d", 10], ~"00010");
    test(#ifmt[~"%.5d", -10], ~"-00010");
    test(#ifmt[~"%.5u", 10u], ~"00010");
    test(#ifmt[~"%.5s", ~"test"], ~"test");
    test(#ifmt[~"%.5x", 127u], ~"0007f");
    test(#ifmt[~"%.5o", 10u], ~"00012");
    test(#ifmt[~"%.5t", 3u], ~"00011");
    test(#ifmt[~"%.5c", 'A'], ~"A");
    // Bool precision. I'm not sure if it's good or bad to have bool
    // conversions support precision - it's not standard printf so we
    // can do whatever. For now I'm making it behave the same as string
    // conversions.

    test(#ifmt[~"%.b", true], ~"");
    test(#ifmt[~"%.0b", true], ~"");
    test(#ifmt[~"%.1b", true], ~"t");
}

fn part5() {
    // Explicit + sign. Only for signed conversions

    test(#ifmt[~"%+d", 0], ~"+0");
    test(#ifmt[~"%+d", 1], ~"+1");
    test(#ifmt[~"%+d", -1], ~"-1");
    // Leave space for sign

    test(#ifmt[~"% d", 0], ~" 0");
    test(#ifmt[~"% d", 1], ~" 1");
    test(#ifmt[~"% d", -1], ~"-1");
    // Plus overrides space

    test(#ifmt[~"% +d", 0], ~"+0");
    test(#ifmt[~"%+ d", 0], ~"+0");
    // 0-padding

    test(#ifmt[~"%05d", 0], ~"00000");
    test(#ifmt[~"%05d", 1], ~"00001");
    test(#ifmt[~"%05d", -1], ~"-0001");
    test(#ifmt[~"%05u", 1u], ~"00001");
    test(#ifmt[~"%05x", 127u], ~"0007f");
    test(#ifmt[~"%05X", 127u], ~"0007F");
    test(#ifmt[~"%05o", 10u], ~"00012");
    test(#ifmt[~"%05t", 3u], ~"00011");
    // 0-padding a string is undefined but glibc does this:

    test(#ifmt[~"%05s", ~"test"], ~" test");
    test(#ifmt[~"%05c", 'A'], ~"    A");
    test(#ifmt[~"%05b", true], ~" true");
    // Left-justify overrides 0-padding

    test(#ifmt[~"%-05d", 0], ~"0    ");
    test(#ifmt[~"%-05d", 1], ~"1    ");
    test(#ifmt[~"%-05d", -1], ~"-1   ");
    test(#ifmt[~"%-05u", 1u], ~"1    ");
    test(#ifmt[~"%-05x", 127u], ~"7f   ");
    test(#ifmt[~"%-05X", 127u], ~"7F   ");
    test(#ifmt[~"%-05o", 10u], ~"12   ");
    test(#ifmt[~"%-05t", 3u], ~"11   ");
    test(#ifmt[~"%-05s", ~"test"], ~"test ");
    test(#ifmt[~"%-05c", 'A'], ~"A    ");
    test(#ifmt[~"%-05b", true], ~"true ");
}
fn part6() {
    // Precision overrides 0-padding

    test(#ifmt[~"%06.5d", 0], ~" 00000");
    test(#ifmt[~"%06.5u", 0u], ~" 00000");
    test(#ifmt[~"%06.5x", 0u], ~" 00000");
    test(#ifmt[~"%06.5d", 10], ~" 00010");
    test(#ifmt[~"%06.5d", -10], ~"-00010");
    test(#ifmt[~"%06.5u", 10u], ~" 00010");
    test(#ifmt[~"%06.5s", ~"test"], ~"  test");
    test(#ifmt[~"%06.5c", 'A'], ~"     A");
    test(#ifmt[~"%06.5x", 127u], ~" 0007f");
    test(#ifmt[~"%06.5X", 127u], ~" 0007F");
    test(#ifmt[~"%06.5o", 10u], ~" 00012");
    // Signed combinations

    test(#ifmt[~"% 5d", 1], ~"    1");
    test(#ifmt[~"% 5d", -1], ~"   -1");
    test(#ifmt[~"%+5d", 1], ~"   +1");
    test(#ifmt[~"%+5d", -1], ~"   -1");
    test(#ifmt[~"% 05d", 1], ~" 0001");
    test(#ifmt[~"% 05d", -1], ~"-0001");
    test(#ifmt[~"%+05d", 1], ~"+0001");
    test(#ifmt[~"%+05d", -1], ~"-0001");
    test(#ifmt[~"%- 5d", 1], ~" 1   ");
    test(#ifmt[~"%- 5d", -1], ~"-1   ");
    test(#ifmt[~"%-+5d", 1], ~"+1   ");
    test(#ifmt[~"%-+5d", -1], ~"-1   ");
    test(#ifmt[~"%- 05d", 1], ~" 1   ");
    test(#ifmt[~"%- 05d", -1], ~"-1   ");
    test(#ifmt[~"%-+05d", 1], ~"+1   ");
    test(#ifmt[~"%-+05d", -1], ~"-1   ");
}
