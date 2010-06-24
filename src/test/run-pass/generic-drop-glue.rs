fn f[T](T t) {
  log "dropping";
}

fn main() {
  type r = rec(@int x, @int y);
  auto x = rec(x=@10, y=@12);
  f[r](x);
}