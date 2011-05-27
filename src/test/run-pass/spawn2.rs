// xfail-stage0
// -*- rust -*-

fn main() {
  spawn child(10, 20, 30, 40, 50, 60, 70, 80, 90);
}

fn child(int i1,
         int i2,
         int i3,
         int i4,
         int i5,
         int i6,
         int i7,
         int i8,
         int i9) 
{
  log_err i1;
  log_err i2;
  log_err i3;
  log_err i4;
  log_err i5;
  log_err i6;
  log_err i7;
  log_err i8;
  log_err i9;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
