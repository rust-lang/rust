extern mod issue_2316_a;

mod cloth {
    #[legacy_exports];

use issue_2316_a::*;

export calico, gingham, flannel;
export fabric;

enum fabric {
  gingham, flannel, calico
}

}


