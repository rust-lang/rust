#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
/sbin/mdev -s

# fill up our entropy pool, if we don't do this then anything with a hash map
# will likely block forever as the kernel is pretty unlikely to have enough
# entropy.
/addentropy < /addentropy
cat /dev/urandom | head -n 2048 | /addentropy

# Set up IP that qemu expects. This configures eth0 with the public IP that QEMU
# will communicate to as well as the loopback 127.0.0.1 address.
ifconfig eth0 10.0.2.15
ifconfig lo up

# Configure DNS resolution of 'localhost' to work
echo 'hosts:      files dns' >> /ubuntu/etc/nsswitch.conf
echo '127.0.0.1    localhost' >> /ubuntu/etc/hosts

# prepare the chroot
mount -t proc proc /ubuntu/proc/
mount --rbind /sys /ubuntu/sys/
mount --rbind /dev /ubuntu/dev/

# Execute our `testd` inside the ubuntu chroot
cp /testd /ubuntu/testd
chroot /ubuntu /testd &
