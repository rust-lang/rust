// -*- rust -*-

// error-pattern: unresolved

tag color {
  rgb(int, int, int);
  rgba(int, int, int, int);
}

fn main() -> () {
  let color red = rgb(255, 0, 0);
  alt (red) {
    case (rgb(?r, ?g, ?b)) {
      log "rgb";
    }
    case (hsl(?h, ?s, ?l)) {
      log "hsl";
    }
  }
}

