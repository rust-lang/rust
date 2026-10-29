#include <assert.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/random.h>

#define N 2048

struct entropy {
  int ent_count;
  int size;
  unsigned char data[N];
};

int main() {
  struct entropy buf;
  ssize_t n;

  int random_fd = open("/dev/random", O_RDWR);
  assert(random_fd >= 0);

  while ((n = read(0, &buf.data, N)) > 0) {
    buf.ent_count = n * 8;
    buf.size = n;
    if (ioctl(random_fd, RNDADDENTROPY, &buf) != 0) {
      perror("failed to add entropy");
    }
  }

  return 0;
}
