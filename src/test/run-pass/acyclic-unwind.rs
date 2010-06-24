// -*- rust -*-

io fn f(chan[int] c)
{
  type t = tup(int,int,int);

  // Allocate an exterior.
  let @t x = tup(1,2,3);

  // Signal parent that we've allocated an exterior.
  c <| 1;

  while (true) {
    // spin waiting for the parent to kill us.
    log "child waiting to die...";
    c <| 1;
  }
}


io fn main() {
  let port[int] p = port();
  spawn f(chan(p));
  let int i;

  // synchronize on event from child.
  i <- p;

  log "parent exiting, killing child";
}
