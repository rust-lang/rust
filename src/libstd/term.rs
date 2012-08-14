//! Simple ANSI color library

import core::option;

// FIXME (#2807): Windows support.

const color_black: u8 = 0u8;
const color_red: u8 = 1u8;
const color_green: u8 = 2u8;
const color_yellow: u8 = 3u8;
const color_blue: u8 = 4u8;
const color_magenta: u8 = 5u8;
const color_cyan: u8 = 6u8;
const color_light_gray: u8 = 7u8;
const color_light_grey: u8 = 7u8;
const color_dark_gray: u8 = 8u8;
const color_dark_grey: u8 = 8u8;
const color_bright_red: u8 = 9u8;
const color_bright_green: u8 = 10u8;
const color_bright_yellow: u8 = 11u8;
const color_bright_blue: u8 = 12u8;
const color_bright_magenta: u8 = 13u8;
const color_bright_cyan: u8 = 14u8;
const color_bright_white: u8 = 15u8;

fn esc(writer: io::Writer) { writer.write(~[0x1bu8, '[' as u8]); }

/// Reset the foreground and background colors to default
fn reset(writer: io::Writer) {
    esc(writer);
    writer.write(~['0' as u8, 'm' as u8]);
}

/// Returns true if the terminal supports color
fn color_supported() -> bool {
    let supported_terms = ~[~"xterm-color", ~"xterm",
                           ~"screen-bce", ~"xterm-256color"];
    return match os::getenv(~"TERM") {
          option::some(env) => {
            for vec::each(supported_terms) |term| {
                if term == env { return true; }
            }
            false
          }
          option::none => false
        };
}

fn set_color(writer: io::Writer, first_char: u8, color: u8) {
    assert (color < 16u8);
    esc(writer);
    let mut color = color;
    if color >= 8u8 { writer.write(~['1' as u8, ';' as u8]); color -= 8u8; }
    writer.write(~[first_char, ('0' as u8) + color, 'm' as u8]);
}

/// Set the foreground color
fn fg(writer: io::Writer, color: u8) {
    return set_color(writer, '3' as u8, color);
}

/// Set the background color
fn bg(writer: io::Writer, color: u8) {
    return set_color(writer, '4' as u8, color);
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
