mat!(t01, r".*abcd", r"abcd", Some((0, 4)));
mat!(t02, r".*(?:abcd)+", r"abcd", Some((0, 4)));
mat!(t03, r".*(?:abcd)+", r"abcdabcd", Some((0, 8)));
mat!(t04, r".*(?:abcd)+", r"abcdxabcd", Some((0, 9)));
mat!(t05, r".*x(?:abcd)+", r"abcdxabcd", Some((0, 9)));
mat!(t06, r"[^abcd]*x(?:abcd)+", r"abcdxabcd", Some((4, 9)));
