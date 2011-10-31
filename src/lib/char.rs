/*
Module: char

Utilities for manipulating the char type
*/

/*
Function: is_whitespace

Indicates whether a character is whitespace.

Whitespace characters include space (U+0020), tab (U+0009), line feed
(U+000A), carriage return (U+000D), and a number of less common
ASCII and unicode characters.
*/
pure fn is_whitespace(c: char) -> bool {
    const ch_space: char = '\u0020';
    const ch_ogham_space_mark: char = '\u1680';
    const ch_mongolian_vowel_sep: char = '\u180e';
    const ch_en_quad: char = '\u2000';
    const ch_em_quad: char = '\u2001';
    const ch_en_space: char = '\u2002';
    const ch_em_space: char = '\u2003';
    const ch_three_per_em_space: char = '\u2004';
    const ch_four_per_em_space: char = '\u2005';
    const ch_six_per_em_space: char = '\u2006';
    const ch_figure_space: char = '\u2007';
    const ch_punctuation_space: char = '\u2008';
    const ch_thin_space: char = '\u2009';
    const ch_hair_space: char = '\u200a';
    const ch_narrow_no_break_space: char = '\u202f';
    const ch_medium_mathematical_space: char = '\u205f';
    const ch_ideographic_space: char = '\u3000';
    const ch_line_separator: char = '\u2028';
    const ch_paragraph_separator: char = '\u2029';
    const ch_character_tabulation: char = '\u0009';
    const ch_line_feed: char = '\u000a';
    const ch_line_tabulation: char = '\u000b';
    const ch_form_feed: char = '\u000c';
    const ch_carriage_return: char = '\u000d';
    const ch_next_line: char = '\u0085';
    const ch_no_break_space: char = '\u00a0';

    if c == ch_space {
        true
    } else if c == ch_ogham_space_mark {
        true
    } else if c == ch_mongolian_vowel_sep {
        true
    } else if c == ch_en_quad {
        true
    } else if c == ch_em_quad {
        true
    } else if c == ch_en_space {
        true
    } else if c == ch_em_space {
        true
    } else if c == ch_three_per_em_space {
        true
    } else if c == ch_four_per_em_space {
        true
    } else if c == ch_six_per_em_space {
        true
    } else if c == ch_figure_space {
        true
    } else if c == ch_punctuation_space {
        true
    } else if c == ch_thin_space {
        true
    } else if c == ch_hair_space {
        true
    } else if c == ch_narrow_no_break_space {
        true
    } else if c == ch_medium_mathematical_space {
        true
    } else if c == ch_ideographic_space {
        true
    } else if c == ch_line_tabulation {
        true
    } else if c == ch_paragraph_separator {
        true
    } else if c == ch_character_tabulation {
        true
    } else if c == ch_line_feed {
        true
    } else if c == ch_line_tabulation {
        true
    } else if c == ch_form_feed {
        true
    } else if c == ch_carriage_return {
        true
    } else if c == ch_next_line {
        true
    } else if c == ch_no_break_space { true } else { false }
}

pure fn to_digit(c: char) -> u8 {
    alt c {
        '0' to '9' { c as u8 - ('0' as u8) }
        'a' to 'z' { c as u8 + 10u8 - ('a' as u8) }
        'A' to 'Z' { c as u8 + 10u8 - ('A' as u8) }
        _ { fail; }
    }
}
