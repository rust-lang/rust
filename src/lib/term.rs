/*
Module: term

Simple ANSI color library
*/

// TODO: Windows support.

/* Const: color_black */
const color_black: u8 = 0u8;
/* Const: color_red */
const color_red: u8 = 1u8;
/* Const: color_green */
const color_green: u8 = 2u8;
/* Const: color_yellow */
const color_yellow: u8 = 3u8;
/* Const: color_blue */
const color_blue: u8 = 4u8;
/* Const: color_magenta */
const color_magenta: u8 = 5u8;
/* Const: color_cyan */
const color_cyan: u8 = 6u8;
/* Const: color_light_gray */
const color_light_gray: u8 = 7u8;
/* Const: color_light_grey */
const color_light_grey: u8 = 7u8;
/* Const: color_dark_gray */
const color_dark_gray: u8 = 8u8;
/* Const: color_dark_grey */
const color_dark_grey: u8 = 8u8;
/* Const: color_bright_red */
const color_bright_red: u8 = 9u8;
/* Const: color_bright_green */
const color_bright_green: u8 = 10u8;
/* Const: color_bright_yellow */
const color_bright_yellow: u8 = 11u8;
/* Const: color_bright_blue */
const color_bright_blue: u8 = 12u8;
/* Const: color_bright_magenta */
const color_bright_magenta: u8 = 13u8;
/* Const: color_bright_cyan */
const color_bright_cyan: u8 = 14u8;
/* Const: color_bright_white */
const color_bright_white: u8 = 15u8;

fn esc(writer: io::buf_writer) { writer.write([0x1bu8, '[' as u8]); }

/*
Function: reset

Reset the foreground and background colors to default
*/
fn reset(writer: io::buf_writer) {
    esc(writer);
    writer.write(['0' as u8, 'm' as u8]);
}

/*
Function: color_supported

Returns true if the terminal supports color
*/
fn color_supported() -> bool {
    let supported_terms = ["xterm-color", "xterm", "screen-bce"];
    ret alt generic_os::getenv("TERM") {
          option::some(env) {
            for term: str in supported_terms {
                if str::eq(term, env) { ret true; }
            }
            false
          }
          option::none. { false }
        };
}

fn set_color(writer: io::buf_writer, first_char: u8, color: u8) {
    assert (color < 16u8);
    esc(writer);
    let color = color;
    if color >= 8u8 { writer.write(['1' as u8, ';' as u8]); color -= 8u8; }
    writer.write([first_char, ('0' as u8) + color, 'm' as u8]);
}

/*
Function: fg

Set the foreground color
*/
fn fg(writer: io::buf_writer, color: u8) {
    ret set_color(writer, '3' as u8, color);
}

/*
Function: fg

Set the background color
*/
fn bg(writer: io::buf_writer, color: u8) {
    ret set_color(writer, '4' as u8, color);
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
