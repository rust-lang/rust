// error-pattern:

/* This is a test to ensure that we do _not_ support nested/balanced comments. I know you might be
   thinking "but nested comments are cool", and that would be a valid point, but they are also a
   thing that would make our lexical syntax non-regular, and we do not want that to be true.

   omitting-things at a higher level (tokens) should be done via token-trees / macros,
   not comments.

   /*
     fail here
   */
*/

fn main() {
}
