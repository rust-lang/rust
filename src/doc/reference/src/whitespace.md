# Whitespace

Whitespace is any non-empty string containing only characters that have the
`Pattern_White_Space` Unicode property, namely:

- `U+0009` (horizontal tab, `'\t'`)
- `U+000A` (line feed, `'\n'`)
- `U+000B` (vertical tab)
- `U+000C` (form feed)
- `U+000D` (carriage return, `'\r'`)
- `U+0020` (space, `' '`)
- `U+0085` (next line)
- `U+200E` (left-to-right mark)
- `U+200F` (right-to-left mark)
- `U+2028` (line separator)
- `U+2029` (paragraph separator)

Rust is a "free-form" language, meaning that all forms of whitespace serve only
to separate _tokens_ in the grammar, and have no semantic significance.

A Rust program has identical meaning if each whitespace element is replaced
with any other legal whitespace element, such as a single space character.
