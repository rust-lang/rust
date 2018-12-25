set -ex

curl -f https://ftp.gnu.org/gnu/make/make-3.81.tar.gz | tar xzf -
cd make-3.81
./configure --prefix=/usr
make
make install
cd ..
rm -rf make-3.81
