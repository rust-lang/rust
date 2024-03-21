mat!(match_flag_case, "(?-u)(?i)abc", "ABC", Some((0, 3)));
mat!(match_flag_weird_case, "(?-u)(?i)a(?-i)bc", "Abc", Some((0, 3)));
mat!(match_flag_weird_case_not, "(?-u)(?i)a(?-i)bc", "ABC", None);
mat!(match_flag_case_dotnl, "(?-u)(?is)a(?u:.)", "A\n", Some((0, 2)));
mat!(
    match_flag_case_dotnl_toggle,
    "(?-u)(?is)a(?u:.)(?-is)a(?u:.)",
    "A\nab",
    Some((0, 4))
);
mat!(
    match_flag_case_dotnl_toggle_not,
    "(?-u)(?is)a(?u:.)(?-is)a(?u:.)",
    "A\na\n",
    None
);
mat!(
    match_flag_case_dotnl_toggle_ok,
    "(?-u)(?is)a(?u:.)(?-is:a(?u:.))?",
    "A\na\n",
    Some((0, 2))
);
mat!(
    match_flag_multi,
    r"(?-u)(?m)(?:^\d+$\n?)+",
    "123\n456\n789",
    Some((0, 11))
);
mat!(match_flag_ungreedy, "(?U)a+", "aa", Some((0, 1)));
mat!(match_flag_ungreedy_greedy, "(?U)a+?", "aa", Some((0, 2)));
mat!(match_flag_ungreedy_noop, "(?U)(?-U)a+", "aa", Some((0, 2)));
