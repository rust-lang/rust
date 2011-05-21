use std;

import std::vec;
import std::option;
import opt = std::getopts;

tag fail_type {
  argument_missing;
  unrecognized_option;
  option_missing;
  option_duplicated;
}

fn check_fail_type(opt::fail_ f, fail_type ft) {
  alt (f) {
    case (opt::argument_missing(_)) {
      assert (ft == argument_missing);
    }
    case (opt::unrecognized_option(_)) {
      assert (ft == unrecognized_option);
    }
    case (opt::option_missing(_)) {
      assert (ft == option_missing);
    }
    case (opt::option_duplicated(_)) {
      assert (ft == option_duplicated);
    }
    case (_) { fail; }
  }
}

// Tests for reqopt

fn test_reqopt_long() {
  auto args = ["--test=20"];
  auto opts = [opt::reqopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "test"));
      assert (opt::opt_str(m, "test") == "20");
    }
    case (_) { fail; }
  }
}

fn test_reqopt_long_missing() {
  auto args = ["blah"];
  auto opts = [opt::reqopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_missing);
    }
    case (_) { fail; }
  }
}

fn test_reqopt_long_no_arg() {
  auto args = ["--test"];
  auto opts = [opt::reqopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, argument_missing);
    }
    case (_) { fail; }
  }
}

fn test_reqopt_long_multi() {
  auto args = ["--test=20", "--test=30"];
  auto opts = [opt::reqopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_duplicated);
    }
    case (_) { fail; }
  }
}

fn test_reqopt_short() {
  auto args = ["-t", "20"];
  auto opts = [opt::reqopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "t"));
      assert (opt::opt_str(m, "t") == "20");
    }
    case (_) { fail; }
  }
}

fn test_reqopt_short_missing() {
  auto args = ["blah"];
  auto opts = [opt::reqopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_missing);
    }
    case (_) { fail; }
  }
}

fn test_reqopt_short_no_arg() {
  auto args = ["-t"];
  auto opts = [opt::reqopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, argument_missing);
    }
    case (_) { fail; }
  }
}

fn test_reqopt_short_multi() {
  auto args = ["-t", "20", "-t", "30"];
  auto opts = [opt::reqopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_duplicated);
    }
    case (_) { fail; }
  }
}


// Tests for optopt

fn test_optopt_long() {
  auto args = ["--test=20"];
  auto opts = [opt::optopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "test"));
      assert (opt::opt_str(m, "test") == "20");
    }
    case (_) { fail; }
  }
}

fn test_optopt_long_missing() {
  auto args = ["blah"];
  auto opts = [opt::optopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (!opt::opt_present(m, "test"));
    }
    case (_) { fail; }
  }
}

fn test_optopt_long_no_arg() {
  auto args = ["--test"];
  auto opts = [opt::optopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, argument_missing);
    }
    case (_) { fail; }
  }
}

fn test_optopt_long_multi() {
  auto args = ["--test=20", "--test=30"];
  auto opts = [opt::optopt("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_duplicated);
    }
    case (_) { fail; }
  }
}

fn test_optopt_short() {
  auto args = ["-t", "20"];
  auto opts = [opt::optopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "t"));
      assert (opt::opt_str(m, "t") == "20");
    }
    case (_) { fail; }
  }
}

fn test_optopt_short_missing() {
  auto args = ["blah"];
  auto opts = [opt::optopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (!opt::opt_present(m, "t"));
    }
    case (_) { fail; }
  }
}

fn test_optopt_short_no_arg() {
  auto args = ["-t"];
  auto opts = [opt::optopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, argument_missing);
    }
    case (_) { fail; }
  }
}

fn test_optopt_short_multi() {
  auto args = ["-t", "20", "-t", "30"];
  auto opts = [opt::optopt("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_duplicated);
    }
    case (_) { fail; }
  }
}


// Tests for optflag

fn test_optflag_long() {
  auto args = ["--test"];
  auto opts = [opt::optflag("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "test"));
    }
    case (_) { fail; }
  }
}

fn test_optflag_long_missing() {
  auto args = ["blah"];
  auto opts = [opt::optflag("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (!opt::opt_present(m, "test"));
    }
    case (_) { fail; }
  }
}

fn test_optflag_long_arg() {
  auto args = ["--test=20"];
  auto opts = [opt::optflag("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) { log_err opt::fail_str(f); }
    case (_) { fail; }
  }
}

fn test_optflag_long_multi() {
  auto args = ["--test", "--test"];
  auto opts = [opt::optflag("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_duplicated);
    }
    case (_) { fail; }
  }
}

fn test_optflag_short() {
  auto args = ["-t"];
  auto opts = [opt::optflag("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "t"));
    }
    case (_) { fail; }
  }
}

fn test_optflag_short_missing() {
  auto args = ["blah"];
  auto opts = [opt::optflag("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (!opt::opt_present(m, "t"));
    }
    case (_) { fail; }
  }
}

fn test_optflag_short_arg() {
  auto args = ["-t", "20"];
  auto opts = [opt::optflag("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      // The next variable after the flag is just a free argument
      assert (m.free.(0) == "20");
    }
    case (_) { fail; }
  }
}

fn test_optflag_short_multi() {
  auto args = ["-t", "-t"];
  auto opts = [opt::optflag("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, option_duplicated);
    }
    case (_) { fail; }
  }
}


// Tests for optmulti

fn test_optmulti_long() {
  auto args = ["--test=20"];
  auto opts = [opt::optmulti("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "test"));
      assert (opt::opt_str(m, "test") == "20");
    }
    case (_) { fail; }
  }
}

fn test_optmulti_long_missing() {
  auto args = ["blah"];
  auto opts = [opt::optmulti("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (!opt::opt_present(m, "test"));
    }
    case (_) { fail; }
  }
}

fn test_optmulti_long_no_arg() {
  auto args = ["--test"];
  auto opts = [opt::optmulti("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, argument_missing);
    }
    case (_) { fail; }
  }
}

fn test_optmulti_long_multi() {
  auto args = ["--test=20", "--test=30"];
  auto opts = [opt::optmulti("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "test"));
      assert (opt::opt_str(m, "test") == "20");
      assert (opt::opt_strs(m, "test").(0) == "20");
      assert (opt::opt_strs(m, "test").(1) == "30");
    }
    case (_) { fail; }
  }
}

fn test_optmulti_short() {
  auto args = ["-t", "20"];
  auto opts = [opt::optmulti("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "t"));
      assert (opt::opt_str(m, "t") == "20");
    }
    case (_) { fail; }
  }
}

fn test_optmulti_short_missing() {
  auto args = ["blah"];
  auto opts = [opt::optmulti("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (!opt::opt_present(m, "t"));
    }
    case (_) { fail; }
  }
}

fn test_optmulti_short_no_arg() {
  auto args = ["-t"];
  auto opts = [opt::optmulti("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, argument_missing);
    }
    case (_) { fail; }
  }
}

fn test_optmulti_short_multi() {
  auto args = ["-t", "20", "-t", "30"];
  auto opts = [opt::optmulti("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (opt::opt_present(m, "t"));
      assert (opt::opt_str(m, "t") == "20");
      assert (opt::opt_strs(m, "t").(0) == "20");
      assert (opt::opt_strs(m, "t").(1) == "30");
    }
    case (_) { fail; }
  }
}

fn test_unrecognized_option_long() {
  auto args = ["--untest"];
  auto opts = [opt::optmulti("t")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, unrecognized_option);
    }
    case (_) { fail; }
  }
}

fn test_unrecognized_option_short() {
  auto args = ["-t"];
  auto opts = [opt::optmulti("test")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::failure(?f)) {
      check_fail_type(f, unrecognized_option);
    }
    case (_) { fail; }
  }
}

fn test_combined() {
  auto args = ["prog", "free1", "-s", "20",
               "free2", "--flag", "--long=30", "-f",
               "-m", "40", "-m", "50"];
  auto opts = [opt::optopt("s"),
               opt::optflag("flag"),
               opt::reqopt("long"),
               opt::optflag("f"),
               opt::optmulti("m"),
               opt::optopt("notpresent")];
  auto res = opt::getopts(args, opts);
  alt (res) {
    case (opt::success(?m)) {
      assert (m.free.(0) == "prog");
      assert (m.free.(1) == "free1");
      assert (opt::opt_str(m, "s") == "20");
      assert (m.free.(2) == "free2");
      assert (opt::opt_present(m, "flag"));
      assert (opt::opt_str(m, "long") == "30");
      assert (opt::opt_present(m, "f"));
      assert (opt::opt_strs(m, "m").(0) == "40");
      assert (opt::opt_strs(m, "m").(1) == "50");
      assert (!opt::opt_present(m, "notpresent"));
    }
    case (_) { fail; }
  }
}

fn main() {

  test_reqopt_long();
  test_reqopt_long_missing();
  test_reqopt_long_no_arg();
  test_reqopt_long_multi();

  test_reqopt_short();
  test_reqopt_short_missing();
  test_reqopt_short_no_arg();
  test_reqopt_short_multi();

  test_optopt_long();
  test_optopt_long_missing();
  test_optopt_long_no_arg();
  test_optopt_long_multi();

  test_optopt_short();
  test_optopt_short_missing();
  test_optopt_short_no_arg();
  test_optopt_short_multi();

  test_optflag_long();
  test_optflag_long_missing();
  // FIXME: Currently long flags will silently accept arguments
  // when it should probably report an error
  //test_optflag_long_arg();
  test_optflag_long_multi();

  test_optflag_short();
  test_optflag_short_missing();
  test_optflag_short_arg();
  test_optflag_short_multi();

  test_optmulti_long();
  test_optmulti_long_missing();
  test_optmulti_long_no_arg();
  test_optmulti_long_multi();

  test_optmulti_short();
  test_optmulti_short_missing();
  test_optmulti_short_no_arg();
  test_optmulti_short_multi();

  test_unrecognized_option_long();
  test_unrecognized_option_short();

  test_combined();
}
