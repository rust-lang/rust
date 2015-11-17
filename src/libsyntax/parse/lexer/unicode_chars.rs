// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Characters and their corresponding confusables were collected from
// http://www.unicode.org/Public/security/revision-06/confusables.txt

use codemap::mk_sp as make_span;
use super::StringReader;

const UNICODE_ARRAY: &'static [(char, &'static str, char)] = &[
    ('ߺ', "Nko Lajanyalan", '_'),
    ('﹍', "Dashed Low Line", '_'),
    ('﹎', "Centreline Low Line", '_'),
    ('﹏', "Wavy Low Line", '_'),
    ('‐', "Hyphen", '-'),
    ('‑', "Non-Breaking Hyphen", '-'),
    ('‒', "Figure Dash", '-'),
    ('–', "En Dash", '-'),
    ('﹘', "Small Em Dash", '-'),
    ('⁃', "Hyphen Bullet", '-'),
    ('˗', "Modifier Letter Minus Sign", '-'),
    ('−', "Minus Sign", '-'),
    ('٫', "Arabic Decimal Separator", ','),
    ('‚', "Single Low-9 Quotation Mark", ','),
    ('ꓹ', "Lisu Letter Tone Na Po", ','),
    (';', "Greek Question Mark", ';'),
    ('ः', "Devanagari Sign Visarga", ':'),
    ('ઃ', "Gujarati Sign Visarga", ':'),
    ('：', "Fullwidth Colon", ':'),
    ('։', "Armenian Full Stop", ':'),
    ('܃', "Syriac Supralinear Colon", ':'),
    ('܄', "Syriac Sublinear Colon", ':'),
    ('︰', "Presentation Form For Vertical Two Dot Leader", ':'),
    ('᠃', "Mongolian Full Stop", ':'),
    ('᠉', "Mongolian Manchu Full Stop", ':'),
    ('⁚', "Two Dot Punctuation", ':'),
    ('׃', "Hebrew Punctuation Sof Pasuq", ':'),
    ('˸', "Modifier Letter Raised Colon", ':'),
    ('꞉', "Modifier Letter Colon", ':'),
    ('∶', "Ratio", ':'),
    ('ː', "Modifier Letter Triangular Colon", ':'),
    ('ꓽ', "Lisu Letter Tone Mya Jeu", ':'),
    ('！', "Fullwidth Exclamation Mark", '!'),
    ('ǃ', "Latin Letter Retroflex Click", '!'),
    ('ʔ', "Latin Letter Glottal Stop", '?'),
    ('ॽ', "Devanagari Letter Glottal Stop", '?'),
    ('Ꭾ', "Cherokee Letter He", '?'),
    ('𝅭', "Musical Symbol Combining Augmentation Dot", '.'),
    ('․', "One Dot Leader", '.'),
    ('۔', "Arabic Full Stop", '.'),
    ('܁', "Syriac Supralinear Full Stop", '.'),
    ('܂', "Syriac Sublinear Full Stop", '.'),
    ('꘎', "Vai Full Stop", '.'),
    ('𐩐', "Kharoshthi Punctuation Dot", '.'),
    ('٠', "Arabic-Indic Digit Zero", '.'),
    ('۰', "Extended Arabic-Indic Digit Zero", '.'),
    ('ꓸ', "Lisu Letter Tone Mya Ti", '.'),
    ('՝', "Armenian Comma", '\''),
    ('＇', "Fullwidth Apostrophe", '\''),
    ('‘', "Left Single Quotation Mark", '\''),
    ('’', "Right Single Quotation Mark", '\''),
    ('‛', "Single High-Reversed-9 Quotation Mark", '\''),
    ('′', "Prime", '\''),
    ('‵', "Reversed Prime", '\''),
    ('՚', "Armenian Apostrophe", '\''),
    ('׳', "Hebrew Punctuation Geresh", '\''),
    ('`', "Greek Varia", '\''),
    ('｀', "Fullwidth Grave Accent", '\''),
    ('΄', "Greek Tonos", '\''),
    ('´', "Greek Oxia", '\''),
    ('᾽', "Greek Koronis", '\''),
    ('᾿', "Greek Psili", '\''),
    ('῾', "Greek Dasia", '\''),
    ('ʹ', "Modifier Letter Prime", '\''),
    ('ʹ', "Greek Numeral Sign", '\''),
    ('ˊ', "Modifier Letter Acute Accent", '\''),
    ('ˋ', "Modifier Letter Grave Accent", '\''),
    ('˴', "Modifier Letter Middle Grave Accent", '\''),
    ('ʻ', "Modifier Letter Turned Comma", '\''),
    ('ʽ', "Modifier Letter Reversed Comma", '\''),
    ('ʼ', "Modifier Letter Apostrophe", '\''),
    ('ʾ', "Modifier Letter Right Half Ring", '\''),
    ('ꞌ', "Latin Small Letter Saltillo", '\''),
    ('י', "Hebrew Letter Yod", '\''),
    ('ߴ', "Nko High Tone Apostrophe", '\''),
    ('ߵ', "Nko Low Tone Apostrophe", '\''),
    ('［', "Fullwidth Left Square Bracket", '('),
    ('❨', "Medium Left Parenthesis Ornament", '('),
    ('❲', "Light Left Tortoise Shell Bracket Ornament", '('),
    ('〔', "Left Tortoise Shell Bracket", '('),
    ('﴾', "Ornate Left Parenthesis", '('),
    ('］', "Fullwidth Right Square Bracket", ')'),
    ('❩', "Medium Right Parenthesis Ornament", ')'),
    ('❳', "Light Right Tortoise Shell Bracket Ornament", ')'),
    ('〕', "Right Tortoise Shell Bracket", ')'),
    ('﴿', "Ornate Right Parenthesis", ')'),
    ('❴', "Medium Left Curly Bracket Ornament", '{'),
    ('❵', "Medium Right Curly Bracket Ornament", '}'),
    ('⁎', "Low Asterisk", '*'),
    ('٭', "Arabic Five Pointed Star", '*'),
    ('∗', "Asterisk Operator", '*'),
    ('᜵', "Philippine Single Punctuation", '/'),
    ('⁁', "Caret Insertion Point", '/'),
    ('∕', "Division Slash", '/'),
    ('⁄', "Fraction Slash", '/'),
    ('╱', "Box Drawings Light Diagonal Upper Right To Lower Left", '/'),
    ('⟋', "Mathematical Rising Diagonal", '/'),
    ('⧸', "Big Solidus", '/'),
    ('㇓', "Cjk Stroke Sp", '/'),
    ('〳', "Vertical Kana Repeat Mark Upper Half", '/'),
    ('丿', "Cjk Unified Ideograph-4E3F", '/'),
    ('⼃', "Kangxi Radical Slash", '/'),
    ('＼', "Fullwidth Reverse Solidus", '\\'),
    ('﹨', "Small Reverse Solidus", '\\'),
    ('∖', "Set Minus", '\\'),
    ('⟍', "Mathematical Falling Diagonal", '\\'),
    ('⧵', "Reverse Solidus Operator", '\\'),
    ('⧹', "Big Reverse Solidus", '\\'),
    ('㇔', "Cjk Stroke D", '\\'),
    ('丶', "Cjk Unified Ideograph-4E36", '\\'),
    ('⼂', "Kangxi Radical Dot", '\\'),
    ('ꝸ', "Latin Small Letter Um", '&'),
    ('﬩', "Hebrew Letter Alternative Plus Sign", '+'),
    ('‹', "Single Left-Pointing Angle Quotation Mark", '<'),
    ('❮', "Heavy Left-Pointing Angle Quotation Mark Ornament", '<'),
    ('˂', "Modifier Letter Left Arrowhead", '<'),
    ('꓿', "Lisu Punctuation Full Stop", '='),
    ('›', "Single Right-Pointing Angle Quotation Mark", '>'),
    ('❯', "Heavy Right-Pointing Angle Quotation Mark Ornament", '>'),
    ('˃', "Modifier Letter Right Arrowhead", '>'),
    ('Ⲻ', "Coptic Capital Letter Dialect-P Ni", '-'),
    ('Ɂ', "Latin Capital Letter Glottal Stop", '?'),
    ('Ⳇ', "Coptic Capital Letter Old Coptic Esh", '/'), ];

const ASCII_ARRAY: &'static [(char, &'static str)] = &[
    ('_', "Underscore"),
    ('-', "Minus/Hyphen"),
    (',', "Comma"),
    (';', "Semicolon"),
    (':', "Colon"),
    ('!', "Exclamation Mark"),
    ('?', "Question Mark"),
    ('.', "Period"),
    ('\'', "Single Quote"),
    ('(', "Left Parenthesis"),
    (')', "Right Parenthesis"),
    ('{', "Left Curly Brace"),
    ('}', "Right Curly Brace"),
    ('*', "Asterisk"),
    ('/', "Slash"),
    ('\\', "Backslash"),
    ('&', "Ampersand"),
    ('+', "Plus Sign"),
    ('<', "Less-Than Sign"),
    ('=', "Equals Sign"),
    ('>', "Greater-Than Sign"), ];

pub fn check_for_substitution(reader: &StringReader, ch: char) {
    UNICODE_ARRAY
    .iter()
    .find(|&&(c, _, _)| c == ch)
    .map(|&(_, u_name, ascii_char)| {
        let span = make_span(reader.last_pos, reader.pos);
        match ASCII_ARRAY.iter().find(|&&(c, _)| c == ascii_char) {
            Some(&(ascii_char, ascii_name)) => {
                let msg =
                    format!("unicode character '{}' ({}) looks much like '{}' ({}), but it's not",
                            ch, u_name, ascii_char, ascii_name);
                reader.help_span(span, &msg);
            },
            None => {
                reader
                .span_diagnostic
                .span_bug_no_panic(span,
                                   &format!("substitution character not found for '{}'", ch));
            }
        }
    });
}
