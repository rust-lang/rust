// -*- rust -*-

type color = tag(
  rgb(int, int, int),
  rgba(int, int, int, int)
);

fn main() -> () {
  let color red = rgb(255, 0, 0);
  alt (red) {
    case (rgb(int r, int g, int b)) {
      log "rgb";
    }
    case (hsl(int h, int s, int l)) {
      log "hsl";
    }
  }
}

