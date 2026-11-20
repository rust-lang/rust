// This won't actually panic because of the error comment -- the `"` needs to be
// the last byte in the file (including not having a trailing newline)
// Prior to the fix you get the error: 'expected item, found `r" ...`'
// because the string being unterminated wasn't properly detected.
r" //~ ERROR unterminated raw string
